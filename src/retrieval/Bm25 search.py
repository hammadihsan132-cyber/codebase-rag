"""
BM25 keyword search, built directly from whatever documents are already
persisted in a repo's Chroma collection (no separate index to keep in sync).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.storage.vector_store import get_or_create_collection

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _tokenize(text: str) -> list[str]:
    """
    Simple code-aware tokenizer: splits identifiers, also breaks
    camelCase/snake_case into sub-tokens so 'getUserData' matches 'user data'.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        tokens.append(raw)
        # split snake_case
        if "_" in raw:
            tokens.extend(p for p in raw.split("_") if p)
    return tokens


@dataclass
class BM25Result:
    chunk_id: str
    content: str
    metadata: dict
    score: float


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]


# Cache is keyed by collection name; call invalidate_bm25_cache() after
# ingesting new data into a repo so the index picks up the fresh chunks.
_index_cache: dict[str, _BM25Index] = {}


def _build_index(owner: str, repo: str) -> _BM25Index:
    collection = get_or_create_collection(owner, repo)
    raw = collection.get(include=["documents", "metadatas"])

    ids = raw["ids"]
    documents = raw["documents"] or []
    metadatas = raw["metadatas"] or []

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else BM25Okapi([[""]])

    return _BM25Index(bm25=bm25, ids=ids, documents=documents, metadatas=metadatas)


def invalidate_bm25_cache(owner: str, repo: str) -> None:
    from src.storage.vector_store import collection_name_for_repo

    _index_cache.pop(collection_name_for_repo(owner, repo), None)


def bm25_search(owner: str, repo: str, query: str, n_results: int = 10) -> list[BM25Result]:
    from src.storage.vector_store import collection_name_for_repo

    key = collection_name_for_repo(owner, repo)
    if key not in _index_cache:
        _index_cache[key] = _build_index(owner, repo)
    index = _index_cache[key]

    if not index.documents:
        return []

    query_tokens = _tokenize(query)
    scores = index.bm25.get_scores(query_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [
        BM25Result(
            chunk_id=index.ids[i],
            content=index.documents[i],
            metadata=index.metadatas[i],
            score=float(scores[i]),
        )
        for i in ranked
        if scores[i] > 0
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    owner, repo, query = (sys.argv[1:4] if len(sys.argv) > 3 else ("psf", "requests-test", "add numbers"))
    results = bm25_search(owner, repo, query)
    for r in results:
        print(f"{r.score:.3f}  {r.metadata.get('file_path')}  {r.content[:60]!r}")