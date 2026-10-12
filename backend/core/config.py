"""
backend/core/config.py
Centralized settings using Pydantic BaseSettings.
All config values come from .env file — nothing hardcoded.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from .env file.
    All values have types — Pydantic validates them at startup.
    If a required value is missing, the app fails fast with a clear error.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str
    qdrant_xray_collection: str = "xray_embeddings"
    qdrant_notes_collection: str = "clinical_notes"
    qdrant_literature_collection: str = "medical_literature"

    # ── HuggingFace ──────────────────────────────────────────────────────────
    huggingface_token: str

    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: str
    gemini_vision_model: str = "gemini-3.5-flash"
    gemini_reasoning_model: str = "gemini-3.5-flash"

    # ── Embedding Models ─────────────────────────────────────────────────────
    pubmedbert_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "openai"

    # ── Data Paths ───────────────────────────────────────────────────────────
    openi_data_dir: str = "data/openi"
    literature_data_dir: str = "data/literature"
    pdf_output_dir: str = "data/pdfs"

    # ── FastAPI ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:4200,http://localhost:3000,http://localhost:5173"

    # ── Ingestion ────────────────────────────────────────────────────────────
    openi_batch_size: int = 32
    rag_top_k: int = 5
    image_max_size_mb: int = 10

    # ── App ──────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    Uses lru_cache so .env is only read once at startup.
    """
    return Settings()
