"""
Persists embedded chunks to ChromaDB using a versioned collection
for NVIDIA Nemotron embeddings.

Supports:
- Deterministic chunk IDs
- Batched upserts
- Repository-level synchronization
- Removal of vectors no longer present in the current successful
  ingestion snapshot
- Vector similarity search
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from config import settings
from src.embedding.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER = "openrouter"
EMBEDDING_MODEL = settings.openrouter_embedding_model
EMBEDDING_DIMENSIONS = 4096
COLLECTION_VERSION = "nemotron_4096"

UPSERT_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 500
READ_BATCH_SIZE = 500

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Initialize the persistent ChromaDB client."""
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )

    return _client


def collection_name_for_repo(owner: str, repo: str) -> str:
    """Generate a versioned collection name for one repository."""
    suffix = f"__{COLLECTION_VERSION}"
    raw = f"{owner}__{repo}"

    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", raw)
    safe = safe.strip("-.")

    if not safe:
        safe = "repo"

    max_prefix_length = 63 - len(suffix)
    prefix = safe[:max_prefix_length].rstrip("-.")

    if len(prefix) < 3:
        prefix = "repo"

    return f"{prefix}{suffix}"


def _chunk_id(embedded: EmbeddedChunk) -> str:
    """
    Generate a deterministic ID.

    Keep this ID scheme stable so existing ChromaDB records can be
    updated in place. Content changes at the same chunk location are
    handled by upsert; chunks that disappear are removed by sync_chunks.
    """
    chunk = embedded.chunk

    raw = (
        f"{chunk.file_path}:"
        f"{chunk.start_line}-{chunk.end_line}:"
        f"{chunk.part or 0}:"
        f"{chunk.symbol_name or ''}"
    )

    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _chunk_metadata(embedded: EmbeddedChunk) -> dict:
    """Build ChromaDB-compatible metadata."""
    chunk = embedded.chunk

    return {
        "file_path": str(chunk.file_path).replace("\\", "/"),
        "language": chunk.language or "text",
        "symbol_name": chunk.symbol_name or "",
        "chunk_type": chunk.chunk_type,
        "start_line": int(chunk.start_line),
        "end_line": int(chunk.end_line),
        "part": int(chunk.part or 0),
        "total_parts": int(chunk.total_parts or 0),
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
    }


def get_or_create_collection(owner: str, repo: str) -> Collection:
    """Get or create the repository's Nemotron collection."""
    client = _get_client()
    name = collection_name_for_repo(owner, repo)

    return client.get_or_create_collection(
        name=name,
        metadata={
            "owner": owner,
            "repo": repo,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "collection_version": COLLECTION_VERSION,
        },
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )


def _validate_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Validate vectors before writing anything to ChromaDB."""
    seen_ids: set[str] = set()

    for item in embedded_chunks:
        if len(item.embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Invalid embedding dimensions: {len(item.embedding)}. "
                f"Expected {EMBEDDING_DIMENSIONS}."
            )

        if not all(math.isfinite(value) for value in item.embedding):
            raise ValueError(
                f"Embedding contains non-finite values for "
                f"{item.chunk.file_path}."
            )

        chunk_id = _chunk_id(item)
        if chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate ChromaDB chunk ID detected: {chunk_id}. "
                "Check the AST chunk IDs and line ranges."
            )
        seen_ids.add(chunk_id)


def _upsert_validated_chunks(
    collection: Collection,
    embedded_chunks: list[EmbeddedChunk],
) -> int:
    """Upsert already-validated chunks and return the number written."""
    if not embedded_chunks:
        return 0

    for start in range(0, len(embedded_chunks), UPSERT_BATCH_SIZE):
        batch = embedded_chunks[start:start + UPSERT_BATCH_SIZE]

        collection.upsert(
            ids=[_chunk_id(item) for item in batch],
            embeddings=[item.embedding for item in batch],
            documents=[item.chunk.content for item in batch],
            metadatas=[_chunk_metadata(item) for item in batch],
        )

        logger.info(
            "Upserted %d/%d chunks in '%s'",
            min(start + len(batch), len(embedded_chunks)),
            len(embedded_chunks),
            collection.name,
        )

    return len(embedded_chunks)


def upsert_chunks(
    owner: str,
    repo: str,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """
    Store embeddings without deleting existing records.

    Use this for non-destructive partial updates. For a completed
    repository ingestion, prefer sync_chunks().
    """
    if not embedded_chunks:
        return

    _validate_embedded_chunks(embedded_chunks)
    collection = get_or_create_collection(owner, repo)
    _upsert_validated_chunks(collection, embedded_chunks)


@dataclass
class SyncResult:
    collection: str
    upserted: int
    deleted: int
    total_after_sync: int


def sync_chunks(
    owner: str,
    repo: str,
    embedded_chunks: list[EmbeddedChunk],
    *,
    allow_empty: bool = False,
) -> SyncResult:
    """
    Synchronize one repository's ChromaDB collection with an ingestion
    snapshot.

    Sequence:
      1. Validate every vector before changing the collection.
      2. Upsert all successfully embedded current chunks.
      3. Read existing IDs in pages.
      4. Delete IDs not present in the current successful snapshot.

    Safety:
      - An empty snapshot is non-destructive unless allow_empty=True.
      - The pipeline should set allow_empty=True only after it has
        accounted for every current chunk (embedded or rejected).
      - If embedding or upsert raises an exception, stale-vector
        deletion is never reached.
    """
    collection = get_or_create_collection(owner, repo)

    if not embedded_chunks and not allow_empty:
        current_count = collection.count()
        logger.warning(
            "Skipping empty synchronization for %s/%s; "
            "preserving %d existing vectors.",
            owner,
            repo,
            current_count,
        )
        return SyncResult(
            collection=collection.name,
            upserted=0,
            deleted=0,
            total_after_sync=current_count,
        )

    _validate_embedded_chunks(embedded_chunks)

    # Upsert first. If this fails, do not proceed to deletion.
    upserted = _upsert_validated_chunks(
        collection,
        embedded_chunks,
    )

    desired_ids = {
        _chunk_id(item)
        for item in embedded_chunks
    }

    # Collect all existing IDs before deleting. This avoids pagination
    # offsets shifting while records are being removed.
    existing_count = collection.count()
    existing_ids: list[str] = []

    for offset in range(0, existing_count, READ_BATCH_SIZE):
        page = collection.get(
            limit=min(READ_BATCH_SIZE, existing_count - offset),
            offset=offset,
            include=["metadatas"],
        )
        existing_ids.extend(page.get("ids", []))

    stale_ids = [
        chunk_id
        for chunk_id in existing_ids
        if chunk_id not in desired_ids
    ]

    deleted = 0

    for start in range(0, len(stale_ids), DELETE_BATCH_SIZE):
        batch_ids = stale_ids[start:start + DELETE_BATCH_SIZE]
        if batch_ids:
            collection.delete(ids=batch_ids)
            deleted += len(batch_ids)

    total_after_sync = collection.count()

    logger.info(
        "ChromaDB sync complete for %s/%s: "
        "upserted=%d, deleted_stale=%d, remaining=%d",
        owner,
        repo,
        upserted,
        deleted,
        total_after_sync,
    )

    return SyncResult(
        collection=collection.name,
        upserted=upserted,
        deleted=deleted,
        total_after_sync=total_after_sync,
    )


@dataclass
class SearchResult:
    content: str
    file_path: str
    symbol_name: str | None
    chunk_type: str
    start_line: int
    end_line: int
    distance: float


def query_collection(
    owner: str,
    repo: str,
    query_embedding: list[float],
    n_results: int = 10,
) -> list[SearchResult]:
    """Search the repository's Nemotron vector collection."""
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Query embedding has {len(query_embedding)} dimensions; "
            f"expected {EMBEDDING_DIMENSIONS}."
        )

    if n_results <= 0:
        return []

    collection = get_or_create_collection(owner, repo)
    total = collection.count()

    if total == 0:
        return []

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    if not raw["ids"] or not raw["ids"][0]:
        return []

    results: list[SearchResult] = []

    for doc, meta, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(
            SearchResult(
                content=doc or "",
                file_path=meta["file_path"],
                symbol_name=meta.get("symbol_name") or None,
                chunk_type=meta["chunk_type"],
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                distance=float(distance),
            )
        )

    return results


def collection_exists(owner: str, repo: str) -> bool:
    """Check whether a Nemotron collection exists."""
    client = _get_client()
    name = collection_name_for_repo(owner, repo)

    return name in {
        collection.name
        for collection in client.list_collections()
    }


def list_ingested_repos() -> list[dict]:
    """List repositories with their embedding information."""
    client = _get_client()
    repos = []

    for collection in client.list_collections():
        metadata = collection.metadata or {}
        owner = metadata.get("owner")
        repo = metadata.get("repo")

        if owner and repo:
            repos.append(
                {
                    "owner": owner,
                    "repo": repo,
                    "collection": collection.name,
                    "embedding_provider": metadata.get(
                        "embedding_provider"
                    ),
                    "embedding_model": metadata.get(
                        "embedding_model"
                    ),
                    "embedding_dimensions": metadata.get(
                        "embedding_dimensions"
                    ),
                    "vector_count": collection.count(),
                }
            )

    return repos


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = _get_client()

    print("\nChromaDB Collections:")
    for collection in client.list_collections():
        print(
            collection.name,
            collection.metadata,
            "vectors=",
            collection.count(),
        )
