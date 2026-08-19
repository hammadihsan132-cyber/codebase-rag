"""
Combines vector similarity search (Chroma) and keyword search (BM25) into
a single ranked result list using Reciprocal Rank Fusion (RRF).

RRF is used instead of raw score blending because vector distances and
BM25 scores live on completely different, incomparable scales — RRF only
needs each ranker's *rank order*, which sidesteps that problem entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.embedding.embedder import embed_query
from src.retrieval.bm25_search import bm25_search
from src.storage.vector_store import query_collection

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF damping constant; higher = flatter weighting across ranks


@dataclass
class RetrievedChunk:
    content: str
    file_path: str
    symbol_name: str | None
    chunk_type: str
    start_line: int
    end_line: int
    rrf_score: float
    matched_by: set[str]  # {"vector"}, {"bm25"}, or {"vector", "bm25"}


def hybrid_search(
    owner: str,
    repo: str,
    query: str,
    n_results: int = 10,
    vector_candidates: int = 20,
    bm25_candidates: int = 20,
) -> list[RetrievedChunk]:
    """
    Run vector search and BM25 search in parallel candidate pools, then
    fuse their rankings. Returns the top `n_results` overall.
    """
    query_embedding = embed_query(query)
    vector_results = query_collection(owner, repo, query_embedding, n_results=vector_candidates)
    keyword_results = bm25_search(owner, repo, query, n_results=bm25_candidates)

    # Vector results don't carry the chunk_id used for BM25 dedup, so key on
    # a (file_path, start_line, end_line) tuple instead — stable across both paths.
    def vec_key(r):
        return (r.file_path, r.start_line, r.end_line)

    def bm25_key(r):
        m = r.metadata
        return (m.get("file_path"), m.get("start_line"), m.get("end_line"))

    fused_scores: dict[tuple, float] = {}
    matched_by: dict[tuple, set[str]] = {}
    payload: dict[tuple, dict] = {}

    for rank, r in enumerate(vector_results, start=1):
        key = vec_key(r)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        matched_by.setdefault(key, set()).add("vector")
        payload[key] = {
            "content": r.content,
            "file_path": r.file_path,
            "symbol_name": r.symbol_name,
            "chunk_type": r.chunk_type,
            "start_line": r.start_line,
            "end_line": r.end_line,
        }

    for rank, r in enumerate(keyword_results, start=1):
        key = bm25_key(r)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        matched_by.setdefault(key, set()).add("bm25")
        if key not in payload:
            m = r.metadata
            payload[key] = {
                "content": r.content,
                "file_path": m.get("file_path"),
                "symbol_name": m.get("symbol_name") or None,
                "chunk_type": m.get("chunk_type"),
                "start_line": m.get("start_line"),
                "end_line": m.get("end_line"),
            }

    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:n_results]

    return [
        RetrievedChunk(
            **payload[key],
            rrf_score=fused_scores[key],
            matched_by=matched_by[key],
        )
        for key in ranked_keys
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    owner, repo, query = (sys.argv[1:4] if len(sys.argv) > 3 else ("psf", "requests", "how does the Session class handle redirects"))
    results = hybrid_search(owner, repo, query)
    for r in results:
        tag = "+".join(sorted(r.matched_by))
        print(f"[{tag:11}] {r.rrf_score:.4f}  {r.file_path} L{r.start_line}-{r.end_line}  {r.chunk_type}:{r.symbol_name}")