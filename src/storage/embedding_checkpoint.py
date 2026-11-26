
"""
Persistent SQLite checkpoint storage for repository embeddings.

Stores successfully generated embeddings and rejected chunks so
repository ingestion can resume without unnecessary API requests.

Features:
- Persistent embedding cache
- Content-based SHA-256 fingerprints
- Rejected chunk tracking
- Provider error tracking
- Rejection attempt counting
- Automatic invalidation when chunk content changes
- Successful embedding removes previous rejection records
- SQLite WAL mode and transactional writes
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import struct
import time

from pathlib import Path

from config import settings
from src.chunking.ast_chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS = 4096
PROVIDER = "openrouter"


class EmbeddingCheckpoint:

    def __init__(
        self,
        db_path: Path | str | None = None,
        dimensions: int = DEFAULT_DIMENSIONS,
    ):
        self.db_path = Path(
            db_path or "./data/checkpoints/embeddings.sqlite3"
        )

        self.dimensions = dimensions
        self.model = settings.openrouter_embedding_model

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )

        connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        return connection

    def _initialize_database(self) -> None:
        """Create checkpoint tables and indexes."""

        connection = self._connect()

        try:
            connection.execute(
                "PRAGMA journal_mode=WAL"
            )

            connection.execute(
                "PRAGMA synchronous=FULL"
            )

            # Successfully generated embeddings
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    fingerprint TEXT PRIMARY KEY,
                    repo_owner TEXT NOT NULL,
                    repo_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_embeddings_repository
                ON embeddings(repo_owner, repo_name)
            """)

            # Provider-rejected chunks
            connection.execute("""
                CREATE TABLE IF NOT EXISTS rejected_chunks (
                    fingerprint TEXT PRIMARY KEY,
                    repo_owner TEXT NOT NULL,
                    repo_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_rejected_repository
                ON rejected_chunks(repo_owner, repo_name)
            """)

            connection.commit()

        finally:
            connection.close()

        logger.info(
            "Embedding checkpoint database initialized: %s",
            self.db_path,
        )

    def fingerprint_chunk(
        self,
        owner: str,
        repo: str,
        chunk: Chunk,
    ) -> str:
        """
        Generate deterministic fingerprints.

        Changes to repository, content, location, chunk structure,
        model, provider or dimensions generate a new fingerprint.
        """

        payload = {
            "owner": owner.lower(),
            "repo": repo.lower(),
            "file_path": str(chunk.file_path).replace("\\", "/"),
            "language": chunk.language,
            "symbol_name": chunk.symbol_name,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "part": chunk.part,
            "total_parts": chunk.total_parts,
            "provider": PROVIDER,
            "model": self.model,
            "dimensions": self.dimensions,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # ---------------------------------------------------------
    # SUCCESSFUL EMBEDDING CACHE
    # ---------------------------------------------------------

    def get_many(
        self,
        fingerprints: list[str],
    ) -> dict[str, list[float]]:
        """Retrieve previously saved embeddings."""

        if not fingerprints:
            return {}

        results: dict[str, list[float]] = {}

        connection = self._connect()

        try:
            for start in range(0, len(fingerprints), 500):

                batch = fingerprints[start:start + 500]

                placeholders = ",".join(
                    "?" for _ in batch
                )

                rows = connection.execute(
                    f"""
                    SELECT fingerprint, dimensions, embedding
                    FROM embeddings
                    WHERE fingerprint IN ({placeholders})
                    AND provider = ?
                    AND model = ?
                    """,
                    (*batch, PROVIDER, self.model),
                ).fetchall()

                for fingerprint, dimensions, blob in rows:

                    if dimensions != self.dimensions:
                        continue

                    expected_bytes = self.dimensions * 4

                    if len(blob) != expected_bytes:
                        logger.warning(
                            "Invalid checkpoint vector: %s",
                            fingerprint,
                        )
                        continue

                    vector = list(
                        struct.unpack(
                            f"<{self.dimensions}f",
                            blob,
                        )
                    )

                    if not all(
                        math.isfinite(value)
                        for value in vector
                    ):
                        logger.warning(
                            "Non-finite checkpoint vector: %s",
                            fingerprint,
                        )
                        continue

                    results[fingerprint] = vector

        finally:
            connection.close()

        logger.info(
            "Checkpoint cache hits: %d/%d",
            len(results),
            len(fingerprints),
        )

        return results

    def save_many(
        self,
        owner: str,
        repo: str,
        items: list[tuple[str, list[float]]],
    ) -> None:
        """
        Save successful embeddings atomically.

        Successful embeddings also remove their previous rejection
        records in the same transaction.
        """

        if not items:
            return

        records = []
        now = time.time()

        for fingerprint, vector in items:

            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {len(vector)}."
                )

            if not all(
                math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    "Embedding contains non-finite values."
                )

            blob = struct.pack(
                f"<{self.dimensions}f",
                *vector,
            )

            records.append((
                fingerprint,
                owner,
                repo,
                PROVIDER,
                self.model,
                self.dimensions,
                blob,
                now,
            ))

        connection = self._connect()

        try:
            connection.executemany(
                """
                INSERT INTO embeddings (
                    fingerprint,
                    repo_owner,
                    repo_name,
                    provider,
                    model,
                    dimensions,
                    embedding,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(fingerprint)
                DO UPDATE SET
                    repo_owner = excluded.repo_owner,
                    repo_name = excluded.repo_name,
                    provider = excluded.provider,
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding,
                    created_at = excluded.created_at
                """,
                records,
            )

            # Remove rejection records after successful embedding.
            connection.executemany(
                """
                DELETE FROM rejected_chunks
                WHERE fingerprint = ?
                """,
                [
                    (fingerprint,)
                    for fingerprint, _ in items
                ],
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        logger.info(
            "Saved %d embeddings to checkpoint database.",
            len(records),
        )

    # ---------------------------------------------------------
    # REJECTED CHUNK MANAGEMENT
    # ---------------------------------------------------------

    def get_rejected(
        self,
        fingerprints: list[str],
    ) -> dict[str, dict]:
        """Retrieve previously rejected chunk information."""

        if not fingerprints:
            return {}

        results: dict[str, dict] = {}

        connection = self._connect()

        try:
            for start in range(0, len(fingerprints), 500):

                batch = fingerprints[start:start + 500]

                placeholders = ",".join(
                    "?" for _ in batch
                )

                rows = connection.execute(
                    f"""
                    SELECT
                        fingerprint,
                        error_message,
                        attempts,
                        created_at,
                        updated_at
                    FROM rejected_chunks
                    WHERE fingerprint IN ({placeholders})
                    AND provider = ?
                    AND model = ?
                    """,
                    (*batch, PROVIDER, self.model),
                ).fetchall()

                for (
                    fingerprint,
                    error,
                    attempts,
                    created,
                    updated,
                ) in rows:

                    results[fingerprint] = {
                        "error": error,
                        "attempts": attempts,
                        "created_at": created,
                        "updated_at": updated,
                    }

        finally:
            connection.close()

        logger.info(
            "Previously rejected chunks: %d/%d",
            len(results),
            len(fingerprints),
        )

        return results

    def mark_rejected_many(
        self,
        owner: str,
        repo: str,
        items: list[tuple[str, str]],
    ) -> None:
        """
        Persist provider-rejected chunks.

        Existing records increment their attempt count.
        """

        if not items:
            return

        now = time.time()

        records = [
            (
                fingerprint,
                owner,
                repo,
                PROVIDER,
                self.model,
                error[:2000],
                now,
                now,
            )
            for fingerprint, error in items
        ]

        connection = self._connect()

        try:
            connection.executemany(
                """
                INSERT INTO rejected_chunks (
                    fingerprint,
                    repo_owner,
                    repo_name,
                    provider,
                    model,
                    error_message,
                    attempts,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)

                ON CONFLICT(fingerprint)
                DO UPDATE SET
                    error_message = excluded.error_message,
                    attempts = rejected_chunks.attempts + 1,
                    updated_at = excluded.updated_at
                """,
                records,
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        logger.info(
            "Recorded %d rejected chunks.",
            len(records),
        )

    def clear_rejected_many(
        self,
        fingerprints: list[str],
    ) -> None:
        """Manually remove rejection records."""

        if not fingerprints:
            return

        connection = self._connect()

        try:
            connection.executemany(
                """
                DELETE FROM rejected_chunks
                WHERE fingerprint = ?
                """,
                [
                    (fingerprint,)
                    for fingerprint in fingerprints
                ],
            )

            connection.commit()

        finally:
            connection.close()

    def count_rejected(
        self,
        owner: str | None = None,
        repo: str | None = None,
    ) -> int:
        """Count rejected chunks."""

        connection = self._connect()

        try:
            if owner is not None and repo is not None:

                result = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM rejected_chunks
                    WHERE repo_owner = ?
                    AND repo_name = ?
                    AND provider = ?
                    AND model = ?
                    """,
                    (
                        owner,
                        repo,
                        PROVIDER,
                        self.model,
                    ),
                ).fetchone()

            else:

                result = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM rejected_chunks
                    WHERE provider = ?
                    AND model = ?
                    """,
                    (PROVIDER, self.model),
                ).fetchone()

            return int(result[0])

        finally:
            connection.close()

    # ---------------------------------------------------------
    # GENERAL STATISTICS
    # ---------------------------------------------------------

    def count(
        self,
        owner: str | None = None,
        repo: str | None = None,
    ) -> int:
        """Count successful embedding checkpoints."""

        connection = self._connect()

        try:
            if owner is not None and repo is not None:

                result = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM embeddings
                    WHERE repo_owner = ?
                    AND repo_name = ?
                    AND provider = ?
                    AND model = ?
                    """,
                    (
                        owner,
                        repo,
                        PROVIDER,
                        self.model,
                    ),
                ).fetchone()

            else:

                result = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM embeddings
                    WHERE provider = ?
                    AND model = ?
                    """,
                    (PROVIDER, self.model),
                ).fetchone()

            return int(result[0])

        finally:
            connection.close()
