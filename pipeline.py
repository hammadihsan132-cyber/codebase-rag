"""
Orchestrates the two end-to-end flows of codebase-rag.

ingest_repository(url):
    clone -> walk -> chunk -> checkpoint lookup
    -> skip known rejections -> embed missing chunks
    -> checkpoint save -> ChromaDB synchronization

answer_question(owner, repo, question):
    hybrid retrieve -> generate cited answer

SQLite checkpoints make embedding ingestion resumable.
ChromaDB synchronization removes vectors that are no longer part
of a successfully accounted-for repository snapshot.
"""

from __future__ import annotations

import argparse
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
from src.storage.vector_store import sync_chunks, upsert_chunks

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
    retry_rejected: bool = False,
) -> IngestResult:
    """
    Complete repository ingestion.

    Flow:
        Clone
        -> Walk
        -> Chunk
        -> Retrieve cached embeddings
        -> Check rejected chunks
        -> Embed eligible chunks
        -> Save checkpoints
        -> Synchronize ChromaDB

    retry_rejected:
        False: Skip unchanged previously rejected chunks.
        True: Retry previously rejected chunks.

    Safety:
        Stale-vector deletion runs only when every current chunk has
        been accounted for as successfully embedded or rejected.
        Unexpected failures leave existing vectors untouched.
    """
    start = time.monotonic()

    # 1. Clone repository.
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

    # 2. Walk repository.
    source_files = walk_repo(repo_info.local_path)

    logger.info(
        "Walked repository: %d files",
        len(source_files),
    )

    # 3. AST-based chunking.
    chunks = chunk_files(source_files)

    logger.info(
        "Created %d chunks",
        len(chunks),
    )

    # Do not interpret an empty walker/chunker result as proof that
    # the repository should be emptied in ChromaDB.
    if not chunks:
        logger.warning(
            "No chunks found for %s. Existing ChromaDB vectors "
            "are being preserved.",
            repo_info.full_name,
        )

        return IngestResult(
            repo=repo_info,
            files_processed=len(source_files),
            chunks_created=0,
            elapsed_seconds=time.monotonic() - start,
        )

    # 4. Initialize SQLite checkpoint.
    checkpoint = EmbeddingCheckpoint()

    fingerprints = [
        checkpoint.fingerprint_chunk(
            owner,
            repo,
            chunk,
        )
        for chunk in chunks
    ]

    # Fingerprints should uniquely identify chunks in this snapshot.
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError(
            "Duplicate chunk fingerprints detected. "
            "Refusing to synchronize ChromaDB."
        )

    # 5. Retrieve successful embeddings.
    cached_embeddings = checkpoint.get_many(fingerprints)

    embedded_by_fingerprint: dict[str, EmbeddedChunk] = {}

    for fingerprint, chunk in zip(fingerprints, chunks):
        vector = cached_embeddings.get(fingerprint)

        if vector is not None:
            embedded_by_fingerprint[fingerprint] = EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
            )

    cache_hits = len(embedded_by_fingerprint)

    # 6. Identify chunks without successful cached embeddings.
    missing_chunks = [
        (fingerprint, chunk)
        for fingerprint, chunk in zip(fingerprints, chunks)
        if fingerprint not in embedded_by_fingerprint
    ]

    # 7. Retrieve previous rejection records.
    missing_fingerprints = [
        fingerprint
        for fingerprint, _ in missing_chunks
    ]

    rejected_records = checkpoint.get_rejected(
        missing_fingerprints
    )

    skipped_rejected: list[tuple[str, object]] = []
    eligible_chunks: list[tuple[str, object]] = []

    for fingerprint, chunk in missing_chunks:
        previous_rejection = rejected_records.get(fingerprint)

        if previous_rejection and not retry_rejected:
            skipped_rejected.append((fingerprint, chunk))
        else:
            eligible_chunks.append((fingerprint, chunk))

    logger.info(
        "Embedding checkpoint summary: "
        "cached=%d, eligible=%d, skipped_rejected=%d",
        cache_hits,
        len(eligible_chunks),
        len(skipped_rejected),
    )

    if retry_rejected:
        logger.info("Manual retry mode enabled.")

    # 8. Embed eligible chunks.
    total_rejected = 0
    newly_embedded_count = 0

    if eligible_chunks:
        embedder = Embedder()

        for start_index in range(
            0,
            len(eligible_chunks),
            EMBEDDING_BATCH_SIZE,
        ):
            batch = eligible_chunks[
                start_index:
                start_index + EMBEDDING_BATCH_SIZE
            ]

            batch_chunks = [
                chunk
                for _, chunk in batch
            ]

            logger.info(
                "Embedding batch %d-%d of %d",
                start_index + 1,
                start_index + len(batch),
                len(eligible_chunks),
            )

            newly_embedded = embedder.embed_chunks(batch_chunks)

            # 9. Match successful results to fingerprints.
            saved_items: list[tuple[str, list[float]]] = []

            for item in newly_embedded:
                fingerprint = checkpoint.fingerprint_chunk(
                    owner,
                    repo,
                    item.chunk,
                )

                embedded_by_fingerprint[fingerprint] = item

                saved_items.append(
                    (
                        fingerprint,
                        item.embedding,
                    )
                )

            # 10. Save successful embeddings immediately.
            if saved_items:
                checkpoint.save_many(
                    owner,
                    repo,
                    saved_items,
                )

                newly_embedded_count += len(saved_items)

                logger.info(
                    "Checkpoint saved: %d embeddings",
                    len(saved_items),
                )

            # 11. Save rejected chunks.
            rejected_items: list[tuple[str, str]] = []

            for failed_chunk in embedder.failed_chunks:
                fingerprint = checkpoint.fingerprint_chunk(
                    owner,
                    repo,
                    failed_chunk,
                )

                error_message = embedder.failed_chunk_errors.get(
                    id(failed_chunk),
                    "Unknown embedding provider error",
                )

                rejected_items.append(
                    (
                        fingerprint,
                        error_message,
                    )
                )

            if rejected_items:
                checkpoint.mark_rejected_many(
                    owner,
                    repo,
                    rejected_items,
                )

                total_rejected += len(rejected_items)

                logger.warning(
                    "Saved %d rejected chunks to SQLite.",
                    len(rejected_items),
                )

    # 12. Restore original chunk order.
    embedded_chunks = [
        embedded_by_fingerprint[fingerprint]
        for fingerprint in fingerprints
        if fingerprint in embedded_by_fingerprint
    ]

    # 13. Confirm that every current chunk has been accounted for.
    #
    # Successful = cached + newly embedded.
    # Rejected = newly rejected + previously rejected and skipped.
    #
    # Only a complete snapshot is allowed to trigger stale-vector
    # deletion.
    accounted_chunks = (
        len(embedded_by_fingerprint)
        + len(skipped_rejected)
        + total_rejected
    )

    coverage_complete = accounted_chunks == len(chunks)

    if not coverage_complete:
        logger.error(
            "Ingestion coverage is incomplete: accounted=%d, "
            "current_chunks=%d. Destructive synchronization "
            "will be skipped.",
            accounted_chunks,
            len(chunks),
        )

    # 14. Ingestion statistics.
    total_rejected_records = checkpoint.count_rejected(
        owner,
        repo,
    )

    logger.info("\n========== EMBEDDING SUMMARY ==========")
    logger.info("Total chunks: %d", len(chunks))
    logger.info("Cached embeddings: %d", cache_hits)
    logger.info("New embeddings: %d", newly_embedded_count)
    logger.info("Rejected this run: %d", total_rejected)
    logger.info(
        "Previously rejected and skipped: %d",
        len(skipped_rejected),
    )
    logger.info(
        "Total stored rejection records: %d",
        total_rejected_records,
    )
    logger.info("Available embeddings: %d", len(embedded_chunks))
    logger.info(
        "Ingestion coverage: %s (%d/%d)",
        "complete" if coverage_complete else "incomplete",
        accounted_chunks,
        len(chunks),
    )
    logger.info("=======================================\n")

    # 15. Synchronize ChromaDB.
    if coverage_complete:
        sync_result = sync_chunks(
            owner,
            repo,
            embedded_chunks,
            allow_empty=True,
        )

        # Refresh keyword retrieval after the vector collection has
        # reached the current repository snapshot.
        invalidate_bm25_cache(owner, repo)

        logger.info(
            "ChromaDB synchronization: collection=%s, "
            "upserted=%d, deleted_stale=%d, remaining=%d",
            sync_result.collection,
            sync_result.upserted,
            sync_result.deleted,
            sync_result.total_after_sync,
        )

    else:
        # A partial result may safely add/update vectors, but must not
        # delete anything because the current snapshot is incomplete.
        if embedded_chunks:
            upsert_chunks(
                owner,
                repo,
                embedded_chunks,
            )

            invalidate_bm25_cache(owner, repo)

            logger.warning(
                "Partial ChromaDB upsert completed. "
                "Stale-vector deletion was skipped."
            )
        else:
            logger.warning(
                "No complete embedding snapshot is available. "
                "Existing ChromaDB vectors were preserved."
            )

    # 16. Return ingestion summary.
    elapsed = time.monotonic() - start

    return IngestResult(
        repo=repo_info,
        files_processed=len(source_files),
        chunks_created=len(chunks),
        elapsed_seconds=elapsed,
    )


# =============================================================
# QUESTION ANSWERING
# =============================================================

def answer_question(
    owner: str,
    repo: str,
    question: str,
    n_results: int = 10,
) -> GeneratedAnswer:
    """Hybrid retrieval followed by cited answer generation."""
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


# =============================================================
# COMMAND LINE INTERFACE
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Codebase RAG ingestion pipeline"
    )

    parser.add_argument(
        "url",
        nargs="?",
        default="https://github.com/psf/requests",
        help="GitHub repository URL",
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force a fresh repository clone",
    )

    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="Retry previously rejected embedding chunks",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    result = ingest_repository(
        url=args.url,
        force_refresh=args.force_refresh,
        retry_rejected=args.retry_rejected,
    )

    print(
        f"\nIngested {result.repo.full_name}: "
        f"{result.files_processed} files -> "
        f"{result.chunks_created} chunks "
        f"in {result.elapsed_seconds:.1f}s"
    )
