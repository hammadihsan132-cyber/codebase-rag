"""
Takes a user's question plus the chunks retrieved for it, and produces a
natural-language answer via Groq — with inline citations back to the
specific files/line-ranges the answer draws from.

Formatting philosophy: this should read like GitHub Copilot Chat / Cursor,
not like a document-RAG summarizer. Concise, scannable, minimal code dumps,
structure only where it earns its keep.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import groq
from groq import APIError, RateLimitError

from config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

_SYSTEM_PROMPT = """You are an expert code intelligence assistant embedded in a developer tool \
(in the spirit of GitHub Copilot Chat, Cursor, or Sourcegraph Cody). You answer questions about \
one specific GitHub repository using ONLY the numbered code excerpts provided to you.

RESPONSE STYLE — non-negotiable:
- Be concise. Prefer short paragraphs and bullet points over long prose. Developers scan, they don't read essays.
- Use markdown headers (##, ###) ONLY when the question actually calls for structure (an overview, an \
architecture explanation). For a narrow question ("what does function X do?"), just answer directly in \
a few sentences — do not force headers onto a two-sentence answer.
- When showing code, show ONLY the minimal relevant excerpt (a function, a class, a handful of lines) — \
never reproduce an entire file. Trim to what's needed to make the point.
- If the question is broad and unscoped (e.g. "show me all the code", "give me the whole project", \
"give me the code of this project"), do NOT dump code from every excerpt. Instead, give a short \
project map: list the key files/modules with a one-line purpose each, then ask which part the user \
wants to see in detail.
- Every factual claim about the code must cite its source excerpt inline using [1], [2], etc., placed \
immediately after the claim it supports. If multiple excerpts support one point, cite all: [1][3].
- Do NOT add your own "Sources", "References", or citation list at the end of your answer — the \
application renders citations separately. Cite inline only, then stop.
- If the excerpts don't contain enough information to answer, say so directly instead of guessing or \
padding the answer with generic knowledge.

FORMAT BY QUESTION TYPE (guide, not a rigid template — use judgment):
- "What does X do?" / "How does X work?" -> 2-4 sentences, one small excerpt if it helps, inline citations.
- "Give me an overview" / "Tell me about this project" -> a few short bullet groups (Purpose, \
Architecture, Tech stack, Key components) — bullets, not paragraphs.
- "Show me the code for X" (a specific, named file/function) -> show that excerpt directly, briefly \
explain what it does.
- "Show me all the code" / "give me the whole project" (broad, unscoped) -> do NOT dump code. Give the \
project map described above instead.
"""

_BROAD_DUMP_PATTERNS = [
    r"\ball\s+(the\s+)?code\b",
    r"\bentire\s+(code\s*base|project|repo(sitory)?)\b",
    r"\bwhole\s+(project|repo(sitory)?|code\s*base)\b",
    r"\bcode\s+of\s+this\s+project\b",
    r"\bevery\s+file\b",
    r"\bfull\s+source\s*code\b",
]

_TRAILING_SOURCES_RE = re.compile(
    r"\n{1,3}\**\s*(sources|references)\s*:?\**\s*\n(?:.*\n?)*$",
    re.IGNORECASE,
)


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


def _is_broad_dump_request(question: str) -> bool:
    q = question.lower()
    return any(re.search(pattern, q) for pattern in _BROAD_DUMP_PATTERNS)


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


def _strip_trailing_sources_section(text: str) -> str:
    """Remove a model-added Sources/References footer, if any slipped through."""
    return _TRAILING_SOURCES_RE.sub("", text).rstrip()


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

        if _is_broad_dump_request(question):
            user_prompt += (
                "\n\nNote: this question is broad and unscoped. Per your instructions, do NOT dump "
                "code from all excerpts above — give a short project map (files/modules + one-line "
                "purpose each) and ask what the user wants to zoom into."
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer_text = self._complete_with_retry(messages)
        answer_text = _strip_trailing_sources_section(answer_text)
        used_citations = _filter_cited(answer_text, citations)
        return GeneratedAnswer(answer=answer_text, citations=used_citations)

    def _complete_with_retry(self, messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.1,
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
