"""
Takes a user's question plus the chunks retrieved for it, and produces a
natural-language answer via Groq — with inline citations back to the
specific files/line-ranges the answer draws from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import groq
from groq import APIError, RateLimitError

from config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

_SYSTEM_PROMPT = """You are a code assistant answering questions about a specific GitHub repository.
You are given a set of numbered code excerpts retrieved from the repo, each labeled with its file path and line range.

Rules:
- Answer using ONLY the information in the provided excerpts. If they don't contain enough to answer, say so plainly.
- Every claim about how the code works must cite the excerpt(s) it's based on, using the format [1], [2], etc.
- Prefer being precise and grounded over being comprehensive — don't speculate about code you haven't been shown.
- If multiple excerpts are relevant to one point, cite all of them, e.g. [1][3].
"""


@dataclass
class Citation:
    index: int
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]


def _build_context_block(chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    blocks = []
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        if chunk.symbol_name:
            label += f" ({chunk.chunk_type} {chunk.symbol_name})"
        blocks.append(f"[{i}] {label}\n```\n{chunk.content}\n```")
        citations.append(
            Citation(
                index=i,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol_name=chunk.symbol_name,
            )
        )
    return "\n\n".join(blocks), citations


class AnswerGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._client = groq.Groq(api_key=api_key or settings.groq_api_key)
        self._model = model or settings.generation_model

    def generate(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if not retrieved_chunks:
            return GeneratedAnswer(
                answer="I couldn't find any relevant code for that question in this repository.",
                citations=[],
            )

        context_block, citations = _build_context_block(retrieved_chunks)
        user_prompt = f"Retrieved excerpts:\n\n{context_block}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer_text = self._complete_with_retry(messages)
        used_citations = _filter_cited(answer_text, citations)
        return GeneratedAnswer(answer=answer_text, citations=used_citations)

    def _complete_with_retry(self, messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.1,  # low temperature: we want grounded, consistent answers
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("Groq rate limit hit (attempt %d/%d), backing off %ds", attempt, _MAX_RETRIES, wait)
                import time
                time.sleep(wait)
            except APIError as e:
                last_error = e
                logger.warning("Groq API error (attempt %d/%d): %s", attempt, _MAX_RETRIES, e)
        raise RuntimeError(f"Groq generation failed after {_MAX_RETRIES} retries") from last_error


def _filter_cited(answer_text: str, citations: list[Citation]) -> list[Citation]:
    """Only surface citations the model actually referenced in its answer."""
    return [c for c in citations if f"[{c.index}]" in answer_text]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path

    sample_chunks = [
        RetrievedChunk(
            content="def add(a, b):\n    return a + b",
            file_path="src/foo.py",
            symbol_name="add",
            chunk_type="function",
            start_line=1,
            end_line=2,
            rrf_score=0.03,
            matched_by={"vector", "bm25"},
        )
    ]
    generator = AnswerGenerator()
    result = generator.generate("What does the add function do?", sample_chunks)
    print(result.answer)
    print(result.citations)