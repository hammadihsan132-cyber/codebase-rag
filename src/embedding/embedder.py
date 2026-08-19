"""
Turns Chunks into embedding vectors via Voyage AI's code-specific model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import voyageai
from voyageai.error import RateLimitError

from config import settings
from src.chunking.ast_chunker import Chunk

logger = logging.getLogger(__name__)

# voyage-code-2 accepts up to 128 texts per request and has a per-request
# token ceiling; keeping batches modest avoids hitting either limit.
_BATCH_SIZE = 100
_MAX_RETRIES = 3


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]


class Embedder:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = voyageai.Client(api_key=api_key or settings.voyage_api_key)
        self._model = model or settings.embedding_model

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of Chunks, batching requests and handling rate limits."""
        if not chunks:
            return []

        results: list[EmbeddedChunk] = []
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start: batch_start + _BATCH_SIZE]
            texts = [self._format_for_embedding(c) for c in batch]
            vectors = self._embed_with_retry(texts)
            results.extend(
                EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(batch, vectors)
            )
            logger.info(
                "Embedded %d/%d chunks", min(batch_start + _BATCH_SIZE, len(chunks)), len(chunks)
            )

        return results

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._client.embed(
                    texts, model=self._model, input_type="document"
                )
                return result.embeddings
            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "Voyage rate limit hit (attempt %d/%d), backing off %ds",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Voyage embedding failed after {_MAX_RETRIES} retries") from last_error

    @staticmethod
    def _format_for_embedding(chunk: Chunk) -> str:
        """
        Prepend a bit of structural context (file path, symbol name) to the
        chunk text so the embedding captures more than just the raw code.
        """
        header_bits = [f"# File: {chunk.file_path}"]
        if chunk.symbol_name:
            header_bits.append(f"# {chunk.chunk_type}: {chunk.symbol_name}")
        header = "\n".join(header_bits)
        return f"{header}\n\n{chunk.content}"


def embed_query(query: str, api_key: str | None = None, model: str | None = None) -> list[float]:
    """Embed a single search query. Uses input_type='query' (asymmetric embedding)."""
    client = voyageai.Client(api_key=api_key or settings.voyage_api_key)
    result = client.embed([query], model=model or settings.embedding_model, input_type="query")
    return result.embeddings[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.chunking.ast_chunker import Chunk
    from pathlib import Path

    sample = Chunk(
        file_path=Path("example.py"),
        language="python",
        symbol_name="add",
        chunk_type="function",
        content="def add(a, b):\n    return a + b",
        start_line=1,
        end_line=2,
    )
    embedder = Embedder()
    out = embedder.embed_chunks([sample])
    print(f"Embedded 1 chunk -> vector of length {len(out[0].embedding)}")