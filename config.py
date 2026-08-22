"""
Central configuration for codebase-rag.
Loads settings from environment variables / a .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys ---
    gemini_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""  # optional, raises GitHub API rate limits for private/large repos

    # --- Storage ---
    clone_dir: Path = Path("./data/repos")       # where repos get shallow-cloned
    chroma_persist_dir: Path = Path("./data/chroma")  # ChromaDB persistence dir

    # --- Embedding / generation models ---
    # embedding_model: str = "models/text-embedding-004"
    embedding_model: str = "models/gemini-embedding-001"
    generation_model: str = "openai/gpt-oss-120b"

    # --- Ingestion limits ---
    max_file_size_bytes: int = 1_000_000  # skip files larger than ~1MB (likely generated/binary)

    # --- Chunking ---
    chunk_max_tokens: int = 500       # rough token budget per chunk before splitting further
    chunk_overlap_lines: int = 3      # lines of overlap between adjacent sub-chunks of a split unit

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist yet."""
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()