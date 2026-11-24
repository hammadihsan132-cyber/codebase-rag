
"""
OpenRouter Nemotron embedding engine.

Features:
- Batch embedding
- Passage and query modes
- Exponential backoff
- Automatic isolation of rejected batches
- Explicit reporting of rejected chunks
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    UnprocessableEntityError,
)

from config import settings
from src.chunking.ast_chunker import Chunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32
_MAX_RETRIES = 5
_INITIAL_RETRY_DELAY = 2
_EXPECTED_DIMENSIONS = 4096


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]


class Embedder:

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._client = OpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=60.0,
            max_retries=0,
        )

        self._model = model or settings.openrouter_embedding_model

        # Explicitly track chunks rejected by the provider.
        self.failed_chunks: list[Chunk] = []

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        results: list[EmbeddedChunk] = []
        self.failed_chunks = []

        for start in range(0, len(chunks), _BATCH_SIZE):

            batch = chunks[start:start + _BATCH_SIZE]

            embedded = self._embed_batch_safely(batch)

            results.extend(embedded)

            logger.info(
                "Processed %d/%d chunks; embedded: %d; rejected: %d",
                min(start + len(batch), len(chunks)),
                len(chunks),
                len(results),
                len(self.failed_chunks),
            )

        if self.failed_chunks:
            logger.error(
                "%d chunks were rejected by the embedding provider.",
                len(self.failed_chunks),
            )

            for chunk in self.failed_chunks:
                logger.error(
                    "Rejected chunk: file=%s lines=%s-%s symbol=%s",
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.symbol_name,
                )

        return results

    def _embed_batch_safely(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Recursively divide a rejected batch to isolate problematic chunks.

        A single rejected chunk is recorded, not silently discarded.
        """

        if not chunks:
            return []

        texts = [
            self._format_for_embedding(chunk)
            for chunk in chunks
        ]

        try:
            vectors = self._embed_with_retry(
                texts,
                input_type="passage",
            )

            return [
                EmbeddedChunk(chunk=chunk, embedding=vector)
                for chunk, vector in zip(chunks, vectors)
            ]

        except UnprocessableEntityError:

            if len(chunks) == 1:

                chunk = chunks[0]
                self.failed_chunks.append(chunk)

                logger.error(
                    "Provider rejected individual chunk: %s, lines %s-%s",
                    chunk.file_path,
                    chunk.start_line,
                    chunk.end_line,
                )

                return []

            midpoint = len(chunks) // 2

            logger.warning(
                "Provider rejected batch of %d chunks. "
                "Splitting into smaller batches.",
                len(chunks),
            )

            left = self._embed_batch_safely(chunks[:midpoint])
            right = self._embed_batch_safely(chunks[midpoint:])

            return left + right

    def _embed_with_retry(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:

        if input_type not in {"passage", "query"}:
            raise ValueError(
                "input_type must be 'passage' or 'query'."
            )

        last_error = None

        for attempt in range(1, _MAX_RETRIES + 1):

            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    encoding_format="float",
                    extra_body={
                        "input_type": input_type,
                    },
                )

                ordered_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                vectors = [
                    item.embedding
                    for item in ordered_data
                ]

                if len(vectors) != len(texts):
                    raise RuntimeError(
                        "Embedding API returned an unexpected "
                        "number of vectors."
                    )

                for vector in vectors:
                    if len(vector) != _EXPECTED_DIMENSIONS:
                        raise RuntimeError(
                            f"Expected {_EXPECTED_DIMENSIONS} dimensions, "
                            f"received {len(vector)}."
                        )

                return vectors

            except UnprocessableEntityError:
                # Let the batch isolation mechanism handle this.
                raise

            except RateLimitError as error:
                last_error = error

            except (
                APIConnectionError,
                APITimeoutError,
            ) as error:
                last_error = error

            except APIError as error:

                if (
                    error.status_code is not None
                    and error.status_code >= 500
                ):
                    last_error = error
                else:
                    raise

            if attempt < _MAX_RETRIES:

                wait = min(
                    _INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                    60,
                )

                logger.warning(
                    "Embedding request failed "
                    "(attempt %d/%d). Retrying in %ds.",
                    attempt,
                    _MAX_RETRIES,
                    wait,
                )

                time.sleep(wait)

        raise RuntimeError(
            f"Embedding failed after {_MAX_RETRIES} attempts."
        ) from last_error

    @staticmethod
    def _format_for_embedding(chunk: Chunk) -> str:

        header = f"# File: {chunk.file_path}"

        if chunk.symbol_name:
            header += (
                f"\n# {chunk.chunk_type}: {chunk.symbol_name}"
            )

        return f"{header}\n\n{chunk.content}"


def embed_query(
    query: str,
    api_key: str | None = None,
    model: str | None = None,
) -> list[float]:

    embedder = Embedder(
        api_key=api_key,
        model=model,
    )

    vectors = embedder._embed_with_retry(
        [query],
        input_type="query",
    )

    return vectors[0]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

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
    output = embedder.embed_chunks([sample])

    print("Embedding test successful!")
    print("Vector dimensions:", len(output[0].embedding))
    print("Rejected chunks:", len(embedder.failed_chunks))
