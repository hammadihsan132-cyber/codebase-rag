"""
Walks a cloned repository and yields the source files worth ingesting,
skipping dependency/build/VCS noise and anything too large or binary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Directories we never want to walk into.
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "vendor", "venv", ".venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "target", "out", "bin", "obj",
    ".next", ".nuxt", ".cache", "coverage",
    "site-packages",
}

# Extension -> language name, used later to pick the right tree-sitter grammar.
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

# Extra file types worth indexing as plain text even without a tree-sitter grammar wired up yet.
PLAIN_TEXT_EXTENSIONS = {
    ".md", ".mdx", ".rst", ".txt",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css",
}


@dataclass
class SourceFile:
    path: Path              # absolute path on disk
    relative_path: Path     # path relative to the repo root (for citations/metadata)
    language: str | None    # tree-sitter language name, or None for plain-text files
    size_bytes: int


def _is_excluded_dir(dirname: str) -> bool:
    return dirname in EXCLUDED_DIRS or dirname.startswith(".")


def walk_repo(repo_path: Path) -> list[SourceFile]:
    """
    Recursively walk repo_path, returning every file we consider worth
    ingesting: known source/text extensions, under the size limit, not
    inside an excluded directory.
    """
    repo_path = Path(repo_path)
    results: list[SourceFile] = []

    for dirpath, dirnames, filenames in _walk_pruned(repo_path):
        for filename in filenames:
            file_path = dirpath / filename
            ext = file_path.suffix.lower()

            language = LANGUAGE_BY_EXTENSION.get(ext)
            is_plain_text = ext in PLAIN_TEXT_EXTENSIONS
            if language is None and not is_plain_text:
                continue  # not a type we know how to handle yet

            try:
                size = file_path.stat().st_size
            except OSError:
                continue  # broken symlink or similar, skip

            if size > settings.max_file_size_bytes:
                logger.debug("Skipping %s (%.1f KB, over limit)", file_path, size / 1024)
                continue
            if size == 0:
                continue

            results.append(
                SourceFile(
                    path=file_path,
                    relative_path=file_path.relative_to(repo_path),
                    language=language,
                    size_bytes=size,
                )
            )

    logger.info("Found %d ingestible files under %s", len(results), repo_path)
    return results


def _walk_pruned(repo_path: Path):
    """os.walk-style traversal that prunes excluded directories in place, using pathlib."""
    import os

    for dirpath_str, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]
        yield Path(dirpath_str), dirnames, filenames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = walk_repo(target)
    for f in files[:20]:
        print(f"{f.language or 'text':<12} {f.relative_path}")
    print(f"... {len(files)} files total")