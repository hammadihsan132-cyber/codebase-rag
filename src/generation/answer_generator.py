
"""
Generates repository-specific answers using OpenRouter.

Default model: NVIDIA Nemotron 3.5 Lightning
Advanced model: Qwen4 Max

Features:
- Repository-grounded answers
- Inline source citations
- Reasoning effort control
- Empty-response recovery
- API retry and error handling
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

from config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_MAX_OUTPUT_TOKENS = 4096

_SYSTEM_PROMPT = """You are an expert code intelligence assistant embedded in
a developer tool similar to GitHub Copilot Chat or Cursor.

You answer questions about one specific GitHub repository using ONLY
the numbered code excerpts provided to you.

RESPONSE STYLE:

- Be concise and use simple, clear explanations.
- Prefer short paragraphs and bullet points.
- Use markdown headers only when necessary.
- Show only relevant code excerpts.
- Never reproduce an entire file unnecessarily.
- Do not invent repository functionality.
- Every factual claim about code must cite its source using [1], [2], etc.
- Place citations immediately after the claims they support.
- Do not create a separate Sources or References section.
- If the provided excerpts are insufficient, clearly say so.

QUESTION TYPES:

1. What does X do?
   Give a concise explanation with citations.

2. Explain project architecture.
   Explain key modules, components and their relationships.

3. Show code for X.
   Display the relevant code excerpt and explain it.

4. Show the whole project.
   Provide a project map instead of dumping every file.

5. Debugging questions.
   Identify relevant code, explain the likely issue and
   distinguish verified facts from possible causes.

Always prioritize repository evidence over assumptions.
"""

_BROAD_DUMP_PATTERNS = [
    r"\ball\s+(the\s+)?code\b",
    r"\bentire\s+(codebase|project|repository|repo)\b",
    r"\bwhole\s+(project|repository|repo|codebase)\b",
    r"\bcode\s+of\s+this\s+project\b",
    r"\bevery\s+file\b",
    r"\bfull\s+source\s*code\b",
]

_TRAILING_SOURCES_RE = re.compile(
    r"\n{1,2}\s{0,3}(?:#{1,3}\s*)?"
    r"(?:sources|references)\s*:?\s*\n"
    r".*$",
    re.IGNORECASE | re.DOTALL,
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
    question = question.lower()

    return any(
        re.search(pattern, question)
        for pattern in _BROAD_DUMP_PATTERNS
    )


def _build_context_block(
    chunks: list[RetrievedChunk],
) -> tuple[str, list[Citation]]:

    blocks = []
    citations = []

    for i, chunk in enumerate(chunks, start=1):

        label = (
            f"{chunk.file_path}:"
            f"{chunk.start_line}-{chunk.end_line}"
        )

        if chunk.symbol_name:
            label += (
                f" ({chunk.chunk_type} {chunk.symbol_name})"
            )

        blocks.append(
            f"[{i}] {label}\n"
            f"```\n{chunk.content}\n```"
        )

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
    """Remove model-generated Sources/References footers."""

    return _TRAILING_SOURCES_RE.sub("", text).rstrip()


def _filter_cited(
    answer_text: str,
    citations: list[Citation],
) -> list[Citation]:
    """Return only citations referenced by the answer."""

    return [
        citation
        for citation in citations
        if re.search(
            rf"(?<!\d)\[{citation.index}\](?!\d)",
            answer_text,
        )
    ]


class AnswerGenerator:

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):

        self._client = OpenAI(
            api_key=api_key or settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=120.0,
            max_retries=0,
        )

        self._model = model or settings.openrouter_chat_model

    def generate(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        advanced: bool = False,
    ) -> GeneratedAnswer:

        if not retrieved_chunks:
            return GeneratedAnswer(
                answer=(
                    "I couldn't find any relevant code "
                    "for that question in this repository."
                ),
                citations=[],
            )

        selected_model = (
            settings.openrouter_advanced_model
            if advanced
            else self._model
        )

        context_block, citations = _build_context_block(
            retrieved_chunks
        )

        user_prompt = (
            f"Retrieved repository excerpts:\n\n"
            f"{context_block}\n\n"
            f"Question: {question}"
        )

        if _is_broad_dump_request(question):
            user_prompt += (
                "\n\nThis is a broad question. Do not dump "
                "the entire codebase. Instead, provide a "
                "short project map with relevant files "
                "and their purposes."
            )

        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        answer_text = self._complete_with_retry(
            messages,
            model=selected_model,
        )

        answer_text = _strip_trailing_sources_section(
            answer_text
        )

        used_citations = _filter_cited(
            answer_text,
            citations,
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=used_citations,
        )

    def _complete_with_retry(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> str:

        selected_model = model or self._model
        last_error = None

        # Nemotron uses low reasoning for ordinary questions.
        # Qwen uses high reasoning for advanced questions.
        is_advanced = (
            selected_model == settings.openrouter_advanced_model
        )

        reasoning_effort = (
            "high" if is_advanced else "low"
        )

        for attempt in range(1, _MAX_RETRIES + 1):

            try:

                request_options = {
                    "model": selected_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": _MAX_OUTPUT_TOKENS,
                }

                # First attempt uses the selected reasoning effort.
                # Second attempt omits reasoning controls if the
                # provider returns an empty response.
                if attempt == 1:
                    request_options["extra_body"] = {
                        "reasoning": {
                            "effort": reasoning_effort
                        }
                    }

                response = (
                    self._client.chat.completions.create(
                        **request_options
                    )
                )

                if not response.choices:
                    raise RuntimeError(
                        "OpenRouter returned no completion choices."
                    )

                choice = response.choices[0]
                answer = choice.message.content

                if not answer or not answer.strip():

                    usage = response.usage
                    reasoning_tokens = None

                    if usage and usage.completion_tokens_details:
                        reasoning_tokens = (
                            usage.completion_tokens_details
                            .reasoning_tokens
                        )

                    last_error = RuntimeError(
                        "OpenRouter returned an empty answer."
                    )

                    logger.warning(
                        "Empty response from %s. "
                        "Attempt=%d/%d, finish_reason=%s, "
                        "reasoning_tokens=%s",
                        selected_model,
                        attempt,
                        _MAX_RETRIES,
                        choice.finish_reason,
                        reasoning_tokens,
                    )

                    if attempt < _MAX_RETRIES:
                        time.sleep(2 ** attempt)
                        continue

                    break

                logger.info(
                    "Generated answer using model=%s, "
                    "reasoning=%s, finish_reason=%s",
                    selected_model,
                    reasoning_effort,
                    choice.finish_reason,
                )

                return answer.strip()

            except RateLimitError as error:

                last_error = error
                wait = min(2 ** attempt, 30)

                logger.warning(
                    "OpenRouter rate limit hit. "
                    "Attempt %d/%d. Retrying in %ds.",
                    attempt,
                    _MAX_RETRIES,
                    wait,
                )

            except (
                APIConnectionError,
                APITimeoutError,
            ) as error:

                last_error = error
                wait = min(2 ** attempt, 30)

                logger.warning(
                    "OpenRouter connection error: %s",
                    error,
                )

            except APIError as error:

                if (
                    error.status_code is not None
                    and error.status_code >= 500
                ):
                    last_error = error
                    wait = min(2 ** attempt, 30)

                    logger.warning(
                        "OpenRouter server error: %s",
                        error,
                    )

                else:
                    raise

            except RuntimeError as error:

                last_error = error

                logger.warning(
                    "Generation attempt %d failed: %s",
                    attempt,
                    error,
                )

                wait = min(2 ** attempt, 30)

            if attempt < _MAX_RETRIES:
                time.sleep(wait)

        raise RuntimeError(
            f"OpenRouter generation failed after "
            f"{_MAX_RETRIES} attempts. Last error: {last_error}"
        ) from last_error


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    sample_chunks = [
        RetrievedChunk(
            content=(
                "def add(a, b):\n"
                "    return a + b"
            ),
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

    result = generator.generate(
        "What does the add function do?",
        sample_chunks,
    )

    print("\nGenerated answer:")
    print(result.answer)

    print("\nCitations:")
    print(result.citations)
