"""
Chunks documentation files (README, docs/*.md, etc.) along their natural
structure — markdown headings — instead of blindly splitting by line count
the way ast_chunker's generic text fallback does. A heading-bounded section
("## Installation" through the next "##") makes a far more coherent and
citable retrieval unit than an arbitrary N-line window.

Non-markdown text files (.txt, .json, .yaml, ...) still fall back to the
plain line-based splitter in ast_chunker, since they have no heading
structure to exploit.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.chunking.ast_chunker import Chunk, _chunk_as_text, _split_lines_with_overlap
from src.ingestion.file_walker import SourceFile
from config import settings

logger = logging.getLogger(__name__)

MARKDOWN_EXTENSIONS = {".md", ".mdx", ".rst"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_into_sections(lines: list[str]) -> list[tuple[str | None, int, list[str]]]:
    """
    Split markdown lines into (heading_text, start_line, section_lines) tuples.
    Content before the first heading (if any) gets heading_text=None.
    """
    sections: list[tuple[str | None, int, list[str]]] = []
    current_heading: str | None = None
    current_start = 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            if current_lines:
                sections.append((current_heading, current_start, current_lines))
            current_heading = match.group(2).strip()
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_start, current_lines))

    return sections


def chunk_doc_file(source_file: SourceFile, source_text: str) -> list[Chunk]:
    """Chunk a documentation file. Markdown/rst gets heading-aware splitting; everything else falls back."""
    ext = source_file.path.suffix.lower()
    if ext not in MARKDOWN_EXTENSIONS:
        return _chunk_as_text(source_file, source_text)

    lines = source_text.split("\n")
    sections = _split_into_sections(lines)

    if not sections:
        return _chunk_as_text(source_file, source_text)

    chunks: list[Chunk] = []
    for heading, start_line, section_lines in sections:
        text = "\n".join(section_lines).strip("\n")
        if not text.strip():
            continue

        end_line = start_line + len(section_lines) - 1
        chunk_tokens = max(1, len(text) // 4)

        if chunk_tokens <= settings.chunk_max_tokens:
            chunks.append(
                Chunk(
                    file_path=source_file.relative_path,
                    language=None,
                    symbol_name=heading,
                    chunk_type="doc_section",
                    content=text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        else:
            # a single section is too big (e.g. a long "## API Reference") — split further, keep the heading as context
            parts = _split_lines_with_overlap(
                section_lines, start_line, settings.chunk_max_tokens, settings.chunk_overlap_lines
            )
            total = len(parts)
            for idx, (part_text, part_start, part_end) in enumerate(parts):
                chunks.append(
                    Chunk(
                        file_path=source_file.relative_path,
                        language=None,
                        symbol_name=heading,
                        chunk_type="doc_section",
                        content=part_text,
                        start_line=part_start,
                        end_line=part_end,
                        part=idx + 1,
                        total_parts=total,
                    )
                )

    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("README.md")
    sf = SourceFile(path=path, relative_path=path, language=None, size_bytes=path.stat().st_size)
    text = path.read_text(encoding="utf-8", errors="replace")
    for c in chunk_doc_file(sf, text):
        label = f" ({c.symbol_name})" if c.symbol_name else ""
        part = f" [part {c.part}/{c.total_parts}]" if c.part else ""
        print(f"L{c.start_line}-{c.end_line}{label}{part}")