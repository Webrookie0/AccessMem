from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5-coder:7b"
    ollama_embed_model: str = "nomic-embed-text"

    # JWT
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Storage
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "secure_docs"
    audit_log_path: str = "./audit.log"

    # RAG
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
