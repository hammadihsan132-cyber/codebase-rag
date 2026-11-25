
"""
Persistent SQLite checkpoint storage for repository embeddings.

Stores successfully generated embeddings so interrupted ingestion
can resume without unnecessarily calling the embedding API again.
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
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        connection = self._connect()

        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")

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
        Generate a deterministic fingerprint.

        Changes to content, location, chunk structure, repository,
        model or dimensions produce a different fingerprint.
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
                    """,
                    batch,
                ).fetchall()

                for fingerprint, dimensions, blob in rows:

                    if dimensions != self.dimensions:
                        logger.warning(
                            "Checkpoint dimension mismatch: %s",
                            fingerprint,
                        )
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

                    if not all(math.isfinite(v) for v in vector):
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
        """Persist successfully generated embeddings atomically."""

        if not items:
            return

        records = []

        for fingerprint, vector in items:

            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, "
                    f"received {len(vector)}."
                )

            if not all(math.isfinite(v) for v in vector):
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
                time.time(),
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
                    embedding = excluded.embedding,
                    created_at = excluded.created_at
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
            "Saved %d embeddings to checkpoint database.",
            len(records),
        )

    def count(self, owner: str | None = None, repo: str | None = None) -> int:
        """Return total checkpoint records, optionally for one repository."""

        connection = self._connect()

        try:
            if owner is not None and repo is not None:
                result = connection.execute(
                    """
                    SELECT COUNT(*) FROM embeddings
                    WHERE repo_owner = ? AND repo_name = ?
                    AND model = ?
                    """,
                    (owner, repo, self.model),
                ).fetchone()
            else:
                result = connection.execute(
                    "SELECT COUNT(*) FROM embeddings"
                ).fetchone()

            return int(result[0])

        finally:
            connection.close()
