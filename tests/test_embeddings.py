import pytest

from app.config import get_settings
from app.services import embeddings
from app.services.ollama import OllamaError


def test_embeddings_are_batched_with_matching_output_count(monkeypatch):
    get_settings().embedding_batch_size = 2
    observed = []

    def embed(method, path, *, payload, timeout):
        observed.append(payload["input"])
        return {"embeddings": [[1.0, float(index + 1)] for index, _ in enumerate(payload["input"])]}

    monkeypatch.setattr(embeddings, "request_json", embed)
    assert len(embeddings.embed_texts(["one", "two", "three"])) == 3
    assert observed == [["one", "two"], ["three"]]


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[float("nan")]], [[True]], [[1], [1, 2]]])
def test_invalid_embedding_payloads_fail(vectors, monkeypatch):
    monkeypatch.setattr(embeddings, "request_json", lambda *args, **kwargs: {"embeddings": vectors})
    with pytest.raises(OllamaError):
        embeddings.embed_texts(["one"] if len(vectors) != 2 else ["one", "two"])
