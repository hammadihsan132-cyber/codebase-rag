
"""
Orchestrates the two end-to-end flows of codebase-rag.

ingest_repository(url):
    clone -> walk -> chunk -> checkpoint lookup
    -> embed missing chunks -> checkpoint save -> ChromaDB

answer_question(owner, repo, question):
    hybrid retrieve -> generate cited answer

SQLite checkpoints make embedding ingestion resumable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.chunking.ast_chunker import chunk_files
from src.embedding.embedder import Embedder, EmbeddedChunk
from src.generation.answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
)
from src.ingestion.file_walker import walk_repo
from src.ingestion.github_loader import RepoInfo, clone_repo
from src.retrieval.bm25_search import invalidate_bm25_cache
from src.retrieval.hybrid_retriever import hybrid_search
from src.storage.embedding_checkpoint import EmbeddingCheckpoint
from src.storage.vector_store import upsert_chunks

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = 32


@dataclass
class IngestResult:
    repo: RepoInfo
    files_processed: int
    chunks_created: int
    elapsed_seconds: float


def ingest_repository(
    url: str,
    force_refresh: bool = False,
) -> IngestResult:

    start = time.monotonic()

    # 1. Clone repository
    repo_info = clone_repo(
        url,
        force_refresh=force_refresh,
    )

    owner = repo_info.owner
    repo = repo_info.name

    logger.info(
        "Cloned %s -> %s",
        repo_info.full_name,
        repo_info.local_path,
    )

    # 2. Walk repository
    source_files = walk_repo(repo_info.local_path)

    logger.info(
        "Walked repository: %d files",
        len(source_files),
    )

    # 3. AST-based chunking
    chunks = chunk_files(source_files)

    logger.info(
        "Created %d chunks",
        len(chunks),
    )

    if not chunks:
        logger.warning(
            "No chunks found for %s",
            repo_info.full_name,
        )

        return IngestResult(
            repo=repo_info,
            files_processed=len(source_files),
            chunks_created=0,
            elapsed_seconds=time.monotonic() - start,
        )

    # 4. Initialize persistent checkpoint storage
    checkpoint = EmbeddingCheckpoint()

    fingerprints = [
        checkpoint.fingerprint_chunk(owner, repo, chunk)
        for chunk in chunks
    ]

    # 5. Retrieve existing embeddings
    cached_embeddings = checkpoint.get_many(fingerprints)

    embedded_by_fingerprint: dict[str, EmbeddedChunk] = {}

    # Reuse previously generated embeddings
    for fingerprint, chunk in zip(fingerprints, chunks):

        vector = cached_embeddings.get(fingerprint)

        if vector is not None:
            embedded_by_fingerprint[fingerprint] = EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
            )

    cache_hits = len(embedded_by_fingerprint)

    # Identify chunks requiring new embeddings
    missing_chunks = [
        (fingerprint, chunk)
        for fingerprint, chunk in zip(fingerprints, chunks)
        if fingerprint not in embedded_by_fingerprint
    ]

    logger.info(
        "Embedding checkpoint: %d cached, %d missing",
        cache_hits,
        len(missing_chunks),
    )

    # 6. Generate only missing embeddings
    total_rejected = 0

    if missing_chunks:

        embedder = Embedder()

        for start_index in range(
            0,
            len(missing_chunks),
            EMBEDDING_BATCH_SIZE,
        ):

            batch = missing_chunks[
                start_index:
                start_index + EMBEDDING_BATCH_SIZE
            ]

            batch_fingerprints = [
                item[0] for item in batch
            ]

            batch_chunks = [
                item[1] for item in batch
            ]

            logger.info(
                "Embedding batch %d-%d of %d",
                start_index + 1,
                start_index + len(batch),
                len(missing_chunks),
            )

            # Generate embeddings
            newly_embedded = embedder.embed_chunks(
                batch_chunks
            )

            # Record rejected chunks
            total_rejected += len(embedder.failed_chunks)

            # Match successful results to their fingerprints
            saved_items = []

            for item in newly_embedded:

                fingerprint = checkpoint.fingerprint_chunk(
                    owner,
                    repo,
                    item.chunk,
                )

                embedded_by_fingerprint[fingerprint] = item

                saved_items.append(
                    (fingerprint, item.embedding)
                )

            # 7. Save successful embeddings immediately
            # This happens BEFORE ChromaDB storage.
            if saved_items:

                checkpoint.save_many(
                    owner,
                    repo,
                    saved_items,
                )

                logger.info(
                    "Checkpoint saved: %d embeddings",
                    len(saved_items),
                )

    # 8. Restore original chunk order
    embedded_chunks = [
        embedded_by_fingerprint[fingerprint]
        for fingerprint in fingerprints
        if fingerprint in embedded_by_fingerprint
    ]

    logger.info(
        "Embedding summary: total=%d, cached=%d, "
        "new=%d, rejected=%d",
        len(chunks),
        cache_hits,
        len(embedded_chunks) - cache_hits,
        total_rejected,
    )

    # 9. Store all successful embeddings in ChromaDB
    if embedded_chunks:

        upsert_chunks(
            owner,
            repo,
            embedded_chunks,
        )

        # Refresh keyword search cache
        invalidate_bm25_cache(owner, repo)

        logger.info(
            "Stored %d embeddings in ChromaDB",
            len(embedded_chunks),
        )

    else:

        logger.error(
            "No embeddings available for %s",
            repo_info.full_name,
        )

    # 10. Return ingestion summary
    elapsed = time.monotonic() - start

    return IngestResult(
        repo=repo_info,
        files_processed=len(source_files),
        chunks_created=len(chunks),
        elapsed_seconds=elapsed,
    )


def answer_question(
    owner: str,
    repo: str,
    question: str,
    n_results: int = 10,
) -> GeneratedAnswer:

    retrieved = hybrid_search(
        owner,
        repo,
        question,
        n_results=n_results,
    )

    logger.info(
        "Retrieved %d chunks for question: %r",
        len(retrieved),
        question,
    )

    generator = AnswerGenerator()

    return generator.generate(
        question,
        retrieved,
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
    )

    import sys

    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://github.com/psf/requests"
    )

    result = ingest_repository(url)

    print(
        f"\nIngested {result.repo.full_name}: "
        f"{result.files_processed} files -> "
        f"{result.chunks_created} chunks "
        f"in {result.elapsed_seconds:.1f}s"
    )
