"""
Takes a user's question plus the chunks retrieved for it, and produces a
natural-language answer via Groq — with inline citations back to the
specific files/line-ranges the answer draws from.

Supports multiple AI modes:
- developer
- student
- debug
- viva
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


# -----------------------------
# AI MODE PROMPTS
# -----------------------------


_DEVELOPER_PROMPT = """
You are an expert code intelligence assistant embedded in a developer tool
(in the spirit of GitHub Copilot Chat, Cursor, or Sourcegraph Cody).

You answer questions about one specific GitHub repository using ONLY the
numbered code excerpts provided.

Rules:

- Be concise and technical.
- Prefer short paragraphs and bullet points.
- Explain architecture and implementation details clearly.
- Use markdown headers only when useful.
- Never dump entire files unless specifically requested.
- Show only relevant code snippets.
- Every factual claim about code must include citations like [1], [2].
- Do not invent information not present in the repository.
- If information is missing, say so clearly.
"""


_STUDENT_PROMPT = """
You are a programming tutor helping students understand a GitHub repository.

Your goal is teaching, not only answering.

Rules:

- Explain concepts step-by-step.
- Use beginner-friendly language.
- Explain why the code exists and how parts connect.
- Use simple examples when useful.
- Avoid unnecessary advanced terminology.
- Mention important files and functions.
- Every code-related statement must include citations like [1], [2].
- Only use information from the provided repository excerpts.
"""


_DEBUG_PROMPT = """
You are an expert debugging assistant analyzing a GitHub repository.

Your job is to help find and fix problems.

Rules:

- Identify possible causes of errors.
- Explain the root cause.
- Mention affected files/functions.
- Suggest practical solutions.
- Explain why the solution works.
- Separate problem, cause, and fix clearly.
- Always cite repository evidence using [1], [2].
- Never invent unsupported code.
"""


_VIVA_PROMPT = """
You are a university project examiner helping students prepare for viva.

Analyze the repository and create useful viva preparation material.

Rules:

- Explain important design decisions.
- Generate possible viva questions.
- Provide clear answers.
- Explain architecture, algorithms, and technologies.
- Highlight important implementation details.
- Use citations like [1], [2].
- Only use information from the repository excerpts.
"""


def _get_system_prompt(mode: str) -> str:

    prompts = {
        "developer": _DEVELOPER_PROMPT,
        "student": _STUDENT_PROMPT,
        "debug": _DEBUG_PROMPT,
        "viva": _VIVA_PROMPT,
    }

    return prompts.get(mode, _DEVELOPER_PROMPT)



# -----------------------------
# Constants
# -----------------------------


_BROAD_DUMP_PATTERNS = [
    r"\ball\s+(the\s+)?code\b",
    r"\bentire\s+(codebase|project|repository)\b",
    r"\bwhole\s+(project|repository|codebase)\b",
    r"\bcode\s+of\s+this\s+project\b",
    r"\bevery\s+file\b",
    r"\bfull\s+source\s+code\b",
]


_TRAILING_SOURCES_RE = re.compile(
    r"\n{1,3}(sources|references)\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)



# -----------------------------
# Data Models
# -----------------------------


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



# -----------------------------
# Helpers
# -----------------------------


def _is_broad_dump_request(question: str) -> bool:

    q = question.lower()

    return any(
        re.search(pattern, q)
        for pattern in _BROAD_DUMP_PATTERNS
    )



def _build_context_block(
    chunks: list[RetrievedChunk]
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
                f" ({chunk.chunk_type} "
                f"{chunk.symbol_name})"
            )


        blocks.append(
            f"[{i}] {label}\n"
            f"```\n"
            f"{chunk.content}\n"
            f"```"
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

    return _TRAILING_SOURCES_RE.sub(
        "",
        text
    ).rstrip()



# -----------------------------
# Generator
# -----------------------------


class AnswerGenerator:


    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None
    ):

        self._client = groq.Groq(
            api_key=api_key or settings.groq_api_key
        )

        self._model = (
            model
            or settings.generation_model
        )



    def generate(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        mode: str = "developer",
    ) -> GeneratedAnswer:


        if not retrieved_chunks:

            return GeneratedAnswer(
                answer=(
                    "I couldn't find any relevant code "
                    "for that question in this repository."
                ),
                citations=[],
            )



        context_block, citations = _build_context_block(
            retrieved_chunks
        )



        user_prompt = (
            f"Retrieved excerpts:\n\n"
            f"{context_block}\n\n"
            f"Question: {question}"
        )



        if _is_broad_dump_request(question):

            user_prompt += (

                "\n\nImportant: This request is broad. "
                "Do not dump all code. Provide a project "
                "map with important files/modules instead."

            )



        messages = [

            {
                "role": "system",
                "content": _get_system_prompt(mode)
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ]



        answer_text = self._complete_with_retry(
            messages
        )


        answer_text = _strip_trailing_sources_section(
            answer_text
        )


        used_citations = _filter_cited(
            answer_text,
            citations
        )


        return GeneratedAnswer(
            answer=answer_text,
            citations=used_citations
        )



    def _complete_with_retry(
        self,
        messages: list[dict]
    ) -> str:


        last_error = None


        for attempt in range(
            1,
            _MAX_RETRIES + 1
        ):


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

                logger.warning(
                    "Groq rate limit. Retry %d/%d after %ds",
                    attempt,
                    _MAX_RETRIES,
                    wait
                )


                import time

                time.sleep(wait)



            except APIError as e:


                last_error = e

                logger.warning(
                    "Groq API error: %s",
                    e
                )



        raise RuntimeError(
            f"Groq generation failed after {_MAX_RETRIES} retries"
        ) from last_error




def _filter_cited(
    answer_text: str,
    citations: list[Citation]
) -> list[Citation]:

    return [
        c
        for c in citations
        if f"[{c.index}]" in answer_text
    ]



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )