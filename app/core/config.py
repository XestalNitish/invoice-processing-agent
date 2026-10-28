"""
app/core/config.py
==================
Central settings loaded from environment / .env file.
All secrets are read here; nothing is hard-coded elsewhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]  # project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Vision LLM ──────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "gemini", "local"] = "anthropic"
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    vision_model: str = "claude-claude-sonnet-4-5"           # override per-provider as needed
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg://invoice_user:invoice_pass@localhost:5432/invoice_db",
        alias="DATABASE_URL",
    )
    db_encrypt_at_rest: bool = True

    # ── PII Masking ───────────────────────────────────────────────────────
    pii_masking_enabled: bool = False  # when True, GSTIN/phone masked in logs

    # ── Storage ──────────────────────────────────────────────────────────
    storage_dir: Path = BASE_DIR / "storage"
    raw_images_dir: Path = BASE_DIR / "storage" / "raw_images"
    processed_images_dir: Path = BASE_DIR / "storage" / "processed_images"
    exports_dir: Path = BASE_DIR / "exports"

    # ── Validation thresholds ─────────────────────────────────────────────
    confidence_review_threshold: float = 0.85   # below this → needs_review
    tax_arithmetic_tolerance: float = 1.00      # rupees tolerance on totals
    line_item_sum_tolerance: float = 1.00

    # ── Extraction retries ────────────────────────────────────────────────
    max_extraction_retries: int = 2

    # ── Local LLM (fallback) ──────────────────────────────────────────────
    local_llm_host: str = "127.0.0.1"
    local_llm_port: int = 8088

    @field_validator("storage_dir", "raw_images_dir", "processed_images_dir", "exports_dir", mode="before")
    @classmethod
    def _ensure_path(cls, v: object) -> Path:
        p = Path(str(v))
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
