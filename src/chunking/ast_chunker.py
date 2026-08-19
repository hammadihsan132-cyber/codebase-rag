"""
Splits source files into semantically meaningful chunks for embedding.

Strategy:
  - For supported languages (python/javascript/typescript/tsx), parse with
    tree-sitter and chunk at function/class boundaries.
  - Any chunk that exceeds `settings.chunk_max_tokens` gets split further
    into overlapping sub-chunks so nothing blows up the embedding context.
  - Top-level code that isn't inside a function/class (imports, constants,
    module-level statements) is grouped into "module" chunks the same way.
  - Files with no tree-sitter grammar (markdown, json, yaml, ...) are
    chunked directly with the same line-based splitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tree_sitter import Language, Node, Parser

from config import settings
from src.ingestion.file_walker import SourceFile

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    file_path: Path            # relative path, for citations
    language: str | None
    symbol_name: str | None    # function/class name, if applicable
    chunk_type: str            # "function" | "class" | "method" | "module" | "text"
    content: str
    start_line: int            # 1-indexed, inclusive
    end_line: int               # 1-indexed, inclusive
    part: int | None = None         # sub-chunk index, if a unit was split
    total_parts: int | None = None


# Node types, per language, that we treat as standalone chunkable definitions.
_DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "export_statement"},
    "typescript": {
        "function_declaration", "class_declaration", "export_statement",
        "interface_declaration", "type_alias_declaration", "enum_declaration",
    },
    "tsx": {
        "function_declaration", "class_declaration", "export_statement",
        "interface_declaration", "type_alias_declaration", "enum_declaration",
    },
}

_CHUNK_TYPE_BY_NODE: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "class_definition": "class",
    "class_declaration": "class",
    "decorated_definition": "function",  # refined to "class" below if it wraps a class
    "export_statement": "function",       # refined below based on wrapped declaration
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}


@lru_cache(maxsize=None)
def _get_parser(language: str) -> Parser | None:
    """Build (and cache) a tree-sitter Parser for the given language name."""
    try:
        if language == "python":
            import tree_sitter_python as ts_lang
            lang = Language(ts_lang.language())
        elif language == "javascript":
            import tree_sitter_javascript as ts_lang
            lang = Language(ts_lang.language())
        elif language == "typescript":
            import tree_sitter_typescript as ts_lang
            lang = Language(ts_lang.language_typescript())
        elif language == "tsx":
            import tree_sitter_typescript as ts_lang
            lang = Language(ts_lang.language_tsx())
        else:
            return None
    except ImportError:
        logger.warning("No tree-sitter grammar installed for language=%s", language)
        return None

    return Parser(lang)


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — good enough for chunk-size budgeting."""
    return max(1, len(text) // 4)


def _get_definition_name(node: Node) -> str | None:
    """Best-effort extraction of a definition's name, unwrapping decorators/exports."""
    name_field = node.child_by_field_name("name")
    if name_field is not None:
        return name_field.text.decode("utf-8", errors="replace")

    # decorated_definition (python) / export_statement (js/ts) wrap the real definition
    for child in node.children:
        if child.type in {
            "function_definition", "class_definition",
            "function_declaration", "class_declaration",
            "interface_declaration", "type_alias_declaration", "enum_declaration",
        }:
            inner_name = child.child_by_field_name("name")
            if inner_name is not None:
                return inner_name.text.decode("utf-8", errors="replace")
    return None


def _get_chunk_type(node: Node) -> str:
    chunk_type = _CHUNK_TYPE_BY_NODE.get(node.type, "module")
    if node.type in {"decorated_definition", "export_statement"}:
        for child in node.children:
            if child.type in {"class_definition", "class_declaration"}:
                return "class"
            if child.type in {"function_definition", "function_declaration"}:
                return "function"
    return chunk_type


def _split_lines_with_overlap(
    lines: list[str],
    start_line: int,
    max_tokens: int,
    overlap_lines: int,
) -> list[tuple[str, int, int]]:
    """
    Greedily pack `lines` into chunks under max_tokens, each new chunk
    re-starting `overlap_lines` lines before where the previous one ended.

    Returns a list of (chunk_text, chunk_start_line, chunk_end_line).
    """
    if not lines:
        return []

    results: list[tuple[str, int, int]] = []
    i = 0
    n = len(lines)

    while i < n:
        current: list[str] = []
        tokens = 0
        j = i
        while j < n:
            line_tokens = _estimate_tokens(lines[j])
            if current and tokens + line_tokens > max_tokens:
                break
            current.append(lines[j])
            tokens += line_tokens
            j += 1

        chunk_start = start_line + i
        chunk_end = start_line + j - 1
        results.append(("\n".join(current), chunk_start, chunk_end))

        if j >= n:
            break
        # step forward, but back up by overlap_lines for context continuity
        i = max(i + 1, j - overlap_lines)

    return results


def _emit_unit(
    text: str,
    start_line: int,
    end_line: int,
    file_path: Path,
    language: str | None,
    symbol_name: str | None,
    chunk_type: str,
) -> list[Chunk]:
    """Turn one logical unit (a function, class, or module-level block) into 1+ Chunks."""
    if _estimate_tokens(text) <= settings.chunk_max_tokens:
        return [
            Chunk(
                file_path=file_path,
                language=language,
                symbol_name=symbol_name,
                chunk_type=chunk_type,
                content=text,
                start_line=start_line,
                end_line=end_line,
            )
        ]

    lines = text.split("\n")
    parts = _split_lines_with_overlap(
        lines, start_line, settings.chunk_max_tokens, settings.chunk_overlap_lines
    )
    total = len(parts)
    return [
        Chunk(
            file_path=file_path,
            language=language,
            symbol_name=symbol_name,
            chunk_type=chunk_type,
            content=part_text,
            start_line=part_start,
            end_line=part_end,
            part=idx + 1,
            total_parts=total,
        )
        for idx, (part_text, part_start, part_end) in enumerate(parts)
    ]


def _chunk_with_ast(source_file: SourceFile, source_text: str) -> list[Chunk]:
    parser = _get_parser(source_file.language)
    if parser is None:
        return _chunk_as_text(source_file, source_text)

    source_bytes = source_text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    def_types = _DEFINITION_TYPES.get(source_file.language, set())

    chunks: list[Chunk] = []
    module_lines: list[str] = []
    module_start_line: int | None = None
    lines = source_text.split("\n")

    def flush_module_block(end_line_exclusive: int) -> None:
        nonlocal module_lines, module_start_line
        if module_lines and module_start_line is not None:
            text = "\n".join(module_lines).strip("\n")
            if text.strip():
                chunks.extend(
                    _emit_unit(
                        text,
                        module_start_line,
                        end_line_exclusive - 1,
                        source_file.relative_path,
                        source_file.language,
                        symbol_name=None,
                        chunk_type="module",
                    )
                )
        module_lines = []
        module_start_line = None

    for child in root.children:
        if child.type in def_types:
            flush_module_block(child.start_point[0] + 1)
            name = _get_definition_name(child)
            chunk_type = _get_chunk_type(child)
            text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
            chunks.extend(
                _emit_unit(
                    text,
                    child.start_point[0] + 1,
                    child.end_point[0] + 1,
                    source_file.relative_path,
                    source_file.language,
                    symbol_name=name,
                    chunk_type=chunk_type,
                )
            )
        else:
            if module_start_line is None:
                module_start_line = child.start_point[0] + 1
            # accumulate raw lines covering this node's span
            for line_no in range(child.start_point[0], child.end_point[0] + 1):
                if line_no < len(lines):
                    module_lines.append(lines[line_no])

    flush_module_block(len(lines) + 1)

    if not chunks:
        # parsing produced nothing chunkable (e.g. empty/trivial file) — fall back to plain text
        return _chunk_as_text(source_file, source_text)

    return chunks


def _chunk_as_text(source_file: SourceFile, source_text: str) -> list[Chunk]:
    lines = source_text.split("\n")
    parts = _split_lines_with_overlap(
        lines, 1, settings.chunk_max_tokens, settings.chunk_overlap_lines
    )
    total = len(parts)
    if total <= 1:
        return [
            Chunk(
                file_path=source_file.relative_path,
                language=source_file.language,
                symbol_name=None,
                chunk_type="text",
                content=source_text,
                start_line=1,
                end_line=len(lines),
            )
        ]
    return [
        Chunk(
            file_path=source_file.relative_path,
            language=source_file.language,
            symbol_name=None,
            chunk_type="text",
            content=part_text,
            start_line=part_start,
            end_line=part_end,
            part=idx + 1,
            total_parts=total,
        )
        for idx, (part_text, part_start, part_end) in enumerate(parts)
    ]


def chunk_source_file(source_file: SourceFile) -> list[Chunk]:
    """Read a SourceFile off disk and split it into Chunks."""
    try:
        source_text = source_file.path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read %s: %s", source_file.path, e)
        return []

    if not source_text.strip():
        return []

    if source_file.language in _DEFINITION_TYPES:
        return _chunk_with_ast(source_file, source_text)
    return _chunk_as_text(source_file, source_text)


def chunk_files(source_files: list[SourceFile]) -> list[Chunk]:
    """Chunk a whole list of SourceFiles, e.g. the output of walk_repo()."""
    all_chunks: list[Chunk] = []
    for sf in source_files:
        all_chunks.extend(chunk_source_file(sf))
    logger.info("Produced %d chunks from %d files", len(all_chunks), len(source_files))
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    from src.ingestion.file_walker import walk_repo

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = walk_repo(target)
    chunks = chunk_files(files)
    for c in chunks[:15]:
        label = f"{c.chunk_type}:{c.symbol_name}" if c.symbol_name else c.chunk_type
        part = f" [part {c.part}/{c.total_parts}]" if c.part else ""
        print(f"{c.file_path} L{c.start_line}-{c.end_line} {label}{part} ({_estimate_tokens(c.content)} tok)")
    print(f"... {len(chunks)} chunks total")