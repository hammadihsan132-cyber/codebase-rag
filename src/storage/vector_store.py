
"""
Persists embedded chunks to ChromaDB using a separate,
versioned collection for NVIDIA Nemotron embeddings.

Provides vector similarity search over repository chunks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from config import settings
from src.embedding.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

# Nemotron embedding configuration
EMBEDDING_PROVIDER = "openrouter"
EMBEDDING_MODEL = settings.openrouter_embedding_model
EMBEDDING_DIMENSIONS = 4096
COLLECTION_VERSION = "nemotron_4096"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Initialize persistent ChromaDB client."""
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )

    return _client


def collection_name_for_repo(owner: str, repo: str) -> str:
    """
    Generate a versioned collection name.

    Keeps Nemotron vectors separate from legacy Gemini vectors.
    Chroma collection names must be 3-63 characters.
    """

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
    """Generate deterministic chunk IDs."""

    chunk = embedded.chunk

    raw = (
        f"{chunk.file_path}:"
        f"{chunk.start_line}-{chunk.end_line}:"
        f"{chunk.part or 0}:"
        f"{chunk.symbol_name or ''}"
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


def _chunk_metadata(embedded: EmbeddedChunk) -> dict:
    """Generate ChromaDB-compatible metadata."""

    chunk = embedded.chunk

    return {
        "file_path": str(chunk.file_path),
        "language": chunk.language or "text",
        "symbol_name": chunk.symbol_name or "",
        "chunk_type": chunk.chunk_type,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "part": chunk.part or 0,
        "total_parts": chunk.total_parts or 0,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
    }


def get_or_create_collection(
    owner: str,
    repo: str,
) -> Collection:
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
                "space": "cosine"
            }
        },
    )


def upsert_chunks(
    owner: str,
    repo: str,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Store Nemotron embeddings in ChromaDB."""

    if not embedded_chunks:
        return

    collection = get_or_create_collection(owner, repo)

    batch_size = 100

    for start in range(0, len(embedded_chunks), batch_size):

        batch = embedded_chunks[start:start + batch_size]

        for item in batch:
            if len(item.embedding) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Invalid embedding dimensions: "
                    f"{len(item.embedding)}. "
                    f"Expected {EMBEDDING_DIMENSIONS}."
                )

        collection.upsert(
            ids=[
                _chunk_id(item)
                for item in batch
            ],
            embeddings=[
                item.embedding
                for item in batch
            ],
            documents=[
                item.chunk.content
                for item in batch
            ],
            metadatas=[
                _chunk_metadata(item)
                for item in batch
            ],
        )

        logger.info(
            "Stored %d/%d chunks in '%s'",
            min(start + len(batch), len(embedded_chunks)),
            len(embedded_chunks),
            collection.name,
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

    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(
            SearchResult(
                content=doc,
                file_path=meta["file_path"],
                symbol_name=meta["symbol_name"] or None,
                chunk_type=meta["chunk_type"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                distance=dist,
            )
        )

    return results


def collection_exists(
    owner: str,
    repo: str,
) -> bool:
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
        )
