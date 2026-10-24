"""Batched and validated embeddings through Ollama."""

from __future__ import annotations

import math

from ..config import get_settings
from .ollama import OllamaError, request_json


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty text")
    settings = get_settings()
    embeddings: list[list[float]] = []
    dimension: int | None = None
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]
        data = request_json(
            "POST",
            "/api/embed",
            payload={"model": settings.ollama_embed_model, "input": batch},
            timeout=settings.ollama_embed_timeout,
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise OllamaError("Ollama returned the wrong number of embeddings")
        for vector in vectors:
            if (
                not isinstance(vector, list)
                or not vector
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
                or not any(vector)
            ):
                raise OllamaError("Ollama returned an invalid embedding")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise OllamaError("Ollama returned inconsistent embedding dimensions")
            embeddings.append([float(value) for value in vector])
    return embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
