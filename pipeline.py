"""
Orchestrates the two end-to-end flows of codebase-rag:

  ingest_repository(url)      -> clone, walk, chunk, embed, store
  answer_question(repo, q)    -> hybrid retrieve, generate cited answer

This is the module both cli.py and api/main.py should call into — neither
should talk to src.* submodules directly, so the pipeline stays the single
source of truth for how the steps fit together.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.chunking.ast_chunker import chunk_files
from src.embedding.embedder import Embedder
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from src.ingestion.file_walker import walk_repo
from src.ingestion.github_loader import RepoInfo, clone_repo
from src.retrieval.bm25_search import invalidate_bm25_cache
from src.retrieval.hybrid_retriever import hybrid_search
from src.storage.vector_store import upsert_chunks

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    repo: RepoInfo
    files_processed: int
    chunks_created: int
    elapsed_seconds: float


def ingest_repository(url: str, force_refresh: bool = False) -> IngestResult:
    """
    Full ingestion flow for a GitHub URL: clone -> walk -> chunk -> embed -> store.
    Safe to call again on the same repo later (chunks are upserted, not duplicated).
    """
    start = time.monotonic()

    repo_info = clone_repo(url, force_refresh=force_refresh)
    logger.info("Cloned %s -> %s", repo_info.full_name, repo_info.local_path)

    source_files = walk_repo(repo_info.local_path)
    logger.info("Walked repo: %d ingestible files", len(source_files))

    chunks = chunk_files(source_files)
    logger.info("Chunked into %d chunks", len(chunks))

    if chunks:
        embedder = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        upsert_chunks(repo_info.owner, repo_info.name, embedded_chunks)
        invalidate_bm25_cache(repo_info.owner, repo_info.name)
        logger.info("Stored %d embedded chunks", len(embedded_chunks))
    else:
        logger.warning("No chunks produced for %s — nothing to embed/store", repo_info.full_name)

    elapsed = time.monotonic() - start
    return IngestResult(
        repo=repo_info,
        files_processed=len(source_files),
        chunks_created=len(chunks),
        elapsed_seconds=elapsed,
    )


def answer_question(
    owner: str, repo: str, question: str, n_results: int = 10
) -> GeneratedAnswer:
    """
    Full query flow for an already-ingested repo: hybrid retrieve -> generate
    a cited answer. Raises if nothing relevant is retrieved.
    """
    retrieved = hybrid_search(owner, repo, question, n_results=n_results)
    logger.info("Retrieved %d chunks for question: %r", len(retrieved), question)

    generator = AnswerGenerator()
    return generator.generate(question, retrieved)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/psf/requests"
    result = ingest_repository(url)
    print(
        f"Ingested {result.repo.full_name}: "
        f"{result.files_processed} files -> {result.chunks_created} chunks "
        f"in {result.elapsed_seconds:.1f}s"
    )