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
            
            # Ollama returns "embeddings" (plural) as a list
            embedding_list = data.get("embeddings")
            if not embedding_list or len(embedding_list) == 0:
                raise RuntimeError(
                    f"No embeddings in Ollama response. Response: {data}\n"
                    f"Make sure Ollama is running: ollama serve\n"
                    f"And model is available: ollama pull {model}"
                )
            # Take the first embedding from the list
            embeddings.append(embedding_list[0])
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}\n"
                f"Error: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get embedding from Ollama: {e}")
    
    return embeddings


def embed_query(text: str) -> list[float]:
    """Generate embedding for a single query text."""
    return embed_texts([text])[0]
