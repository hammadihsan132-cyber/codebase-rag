"""
Persists embedded chunks to ChromaDB (one collection per repo) and
provides vector similarity search over them.
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

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return _client


def collection_name_for_repo(owner: str, repo: str) -> str:
    """
    Chroma collection names must be 3-63 chars, start/end alphanumeric,
    and contain only [a-zA-Z0-9._-]. Sanitize owner/repo into a safe name.
    """
    raw = f"{owner}__{repo}"
    safe = re.sub(r"[^a-zA-Z0-9._-]", "-", raw)
    safe = safe.strip("-.")
    if len(safe) < 3:
        safe = f"repo-{safe}"
    return safe[:63]


def _chunk_id(embedded: EmbeddedChunk) -> str:
    """Deterministic ID so re-ingesting the same file/chunk overwrites rather than duplicates."""
    c = embedded.chunk
    raw = f"{c.file_path}:{c.start_line}-{c.end_line}:{c.part or 0}:{c.symbol_name or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _chunk_metadata(embedded: EmbeddedChunk) -> dict:
    c = embedded.chunk
    return {
        "file_path": str(c.file_path),
        "language": c.language or "text",
        "symbol_name": c.symbol_name or "",
        "chunk_type": c.chunk_type,
        "start_line": c.start_line,
        "end_line": c.end_line,
        "part": c.part or 0,
        "total_parts": c.total_parts or 0,
    }


def get_or_create_collection(owner: str, repo: str) -> Collection:
    client = _get_client()
    name = collection_name_for_repo(owner, repo)
    return client.get_or_create_collection(name=name, metadata={"owner": owner, "repo": repo})


def upsert_chunks(owner: str, repo: str, embedded_chunks: list[EmbeddedChunk]) -> None:
    """Write embedded chunks into the repo's collection, in batches."""
    if not embedded_chunks:
        return

    collection = get_or_create_collection(owner, repo)
    batch_size = 500  # Chroma handles large batches fine; keep requests bounded

    for start in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[start: start + batch_size]
        collection.upsert(
            ids=[_chunk_id(e) for e in batch],
            embeddings=[e.embedding for e in batch],
            documents=[e.chunk.content for e in batch],
            metadatas=[_chunk_metadata(e) for e in batch],
        )
        logger.info(
            "Upserted %d/%d chunks into collection '%s'",
            min(start + batch_size, len(embedded_chunks)),
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
    owner: str, repo: str, query_embedding: list[float], n_results: int = 10
) -> list[SearchResult]:
    """Vector similarity search against a repo's collection."""
    collection = get_or_create_collection(owner, repo)
    raw = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    if not raw["ids"] or not raw["ids"][0]:
        return []

    results: list[SearchResult] = []
    for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
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


def collection_exists(owner: str, repo: str) -> bool:
    client = _get_client()
    name = collection_name_for_repo(owner, repo)
    return name in {c.name for c in client.list_collections()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Existing collections:", [c.name for c in _get_client().list_collections()])