import pytest
import requests

from app.services import ollama
from app.services.ollama import OllamaError


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.mark.parametrize("payload", [[], None, {"error": "internal details"}])
def test_invalid_json_shapes_are_rejected(payload, monkeypatch):
    monkeypatch.setattr(ollama._session(), "request", lambda *args, **kwargs: Response(payload))
    with pytest.raises(OllamaError):
        ollama.request_json("POST", "/api/chat", payload={}, timeout=1)


def test_connection_errors_are_wrapped_without_private_details(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("secret internal URL")

    monkeypatch.setattr(ollama._session(), "request", fail)
    with pytest.raises(OllamaError) as error:
        ollama.request_json("POST", "/api/chat", payload={}, timeout=1)
    assert "secret" not in str(error.value)
