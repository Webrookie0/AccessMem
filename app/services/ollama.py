"""Connection pooling and bounded, validated requests to the model server."""

from __future__ import annotations

from threading import local

import requests

from ..config import get_settings

_local = local()


class OllamaError(RuntimeError):
    """The model server is unavailable or returned an invalid response."""


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def request_json(method: str, path: str, *, payload: dict | None = None, timeout: float) -> dict:
    url = get_settings().ollama_base_url.rstrip("/") + path
    try:
        response = _session().request(method, url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError("Ollama is unavailable or returned an invalid response.") from exc
    if not isinstance(data, dict) or data.get("error"):
        raise OllamaError("Ollama returned an invalid response.")
    return data


def available_models() -> set[str]:
    data = request_json("GET", "/api/tags", timeout=5)
    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama returned an invalid model list.")
    return {
        model["name"]
        for model in models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }
