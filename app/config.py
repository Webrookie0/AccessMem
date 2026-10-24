"""Validated configuration for AccessMem."""

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5-coder:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_timeout: float = Field(default=120, gt=0)
    ollama_embed_timeout: float = Field(default=60, gt=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # A development process gets a random secret if no .env has been created.
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(48), repr=False)
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_issuer: str = "accessmem"
    jwt_audience: str = "accessmem-api"
    jwt_expire_minutes: int = Field(default=60, ge=1, le=1440)
    demo_users_enabled: bool = True
    user_database_path: str = "./accessmem_users.db"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "secure_docs"
    audit_log_path: str = "./audit.log"
    audit_log_max_bytes: int = Field(default=5_000_000, ge=1024)
    audit_log_backup_count: int = Field(default=3, ge=1, le=20)
    audit_log_questions: bool = False
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    login_requests_per_minute: int = Field(default=10, ge=1)
    chat_requests_per_minute: int = Field(default=30, ge=1)

    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=4, ge=1, le=100)
    retrieval_candidates: int = Field(default=20, ge=1, le=500)
    rrf_rank_constant: int = Field(default=60, ge=1)
    minimum_similarity: float = Field(default=0.2, ge=-1, le=1)
    chunk_chars: int = Field(default=800, ge=100, le=20000)
    chunk_overlap: int = Field(default=100, ge=0)
    ingest_corpus_id: str = Field(default="seed", min_length=1)

    # Evaluation uses Ollama's OpenAI-compatible API, independently of generation.
    ragas_base_url: str = "http://localhost:11434/v1"
    ragas_model: str = "qwen2.5-coder:7b"
    ragas_embed_model: str = "nomic-embed-text"
    ragas_api_key: str = Field(default="ollama", repr=False)
    ragas_timeout: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_CHARS")
        if self.app_env == "production":
            if len(self.jwt_secret) < 32 or self.jwt_secret.startswith("change-me"):
                raise ValueError(
                    "Production JWT_SECRET must be a random secret of at least 32 characters"
                )
            if "jwt_secret" not in self.model_fields_set:
                raise ValueError("Production requires an explicitly configured JWT_SECRET")
            if self.demo_users_enabled:
                raise ValueError(
                    "Set DEMO_USERS_ENABLED=false in production and provision real accounts"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
