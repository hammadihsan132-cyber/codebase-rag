
"""
Central configuration for Codebase RAG.
Loads settings from environment variables / a .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API keys ---
    openrouter_api_key: str = ""

    # Keep existing keys temporarily for migration compatibility
    gemini_api_key: str = ""
    groq_api_key: str = ""
    github_token: str = ""

    # --- OpenRouter models ---
    openrouter_embedding_model: str = (
        "nvidia/nemotron-3.5-embed-8b:free"
    )

    openrouter_chat_model: str = (
        "nvidia/nemotron-3.5-lightning:free"
    )

    openrouter_advanced_model: str = (
        "qwen/qwen4-max"
    )

    # --- OpenRouter API ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Legacy model settings ---
    # Retained temporarily to avoid breaking existing code.
    embedding_model: str = "models/gemini-embedding-001"
    generation_model: str = "openai/gpt-oss-120b"

    # --- Storage ---
    clone_dir: Path = Path("./data/repos")
    chroma_persist_dir: Path = Path("./data/chroma")

    # --- Ingestion limits ---
    max_file_size_bytes: int = 1_000_000

    # --- Chunking ---
    chunk_max_tokens: int = 500
    chunk_overlap_lines: int = 3

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
