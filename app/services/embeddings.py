"""Ollama embeddings wrapper."""
import requests
from functools import lru_cache

from ..config import get_settings


def _get_base_url() -> str:
    return get_settings().ollama_base_url.rstrip("/")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama."""
    if not texts:
        return []
    
    base_url = _get_base_url()
    model = get_settings().ollama_embed_model
    
    embeddings = []
    for text in texts:
        try:
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data.get("embedding", []))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get embedding from Ollama: {e}")
    
    return embeddings


def embed_query(text: str) -> list[float]:
    """Generate embedding for a single query text."""
    return embed_texts([text])[0]
