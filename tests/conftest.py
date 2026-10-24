"""Isolated real storage with deterministic embeddings and a model stub."""

import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.rate_limit import limiter
from app.services import vector_store
from app.services.guardrails import REFUSAL


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    values = {
        "APP_ENV": "test",
        "JWT_SECRET": "test-accessmem-secret-that-is-long-and-deterministic",
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "USER_DATABASE_PATH": str(tmp_path / "users.db"),
        "AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        "DEMO_USERS_ENABLED": "true",
        "MINIMUM_SIMILARITY": "0.2",
        "LOGIN_REQUESTS_PER_MINUTE": "100",
        "CHAT_REQUESTS_PER_MINUTE": "100",
        "AUDIT_LOG_QUESTIONS": "false",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    vector_store._client.cache_clear()
    limiter.clear()
    yield
    if vector_store._client.cache_info().currsize:
        vector_store._client()._system.stop()
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    audit_logger = logging.getLogger(f"accessmem.audit.{Path(values['AUDIT_LOG_PATH']).resolve()}")
    for handler in audit_logger.handlers[:]:
        handler.close()
        audit_logger.removeHandler(handler)
    from app.services.audit import _logger

    _logger.cache_clear()
    get_settings.cache_clear()
    vector_store._client.cache_clear()
    limiter.clear()


def fake_embedding(text: str) -> list[float]:
    terms = set(re.findall(r"\w+", text.lower()))
    vocabulary = ("vacation", "northwind", "forecast", "poison", "widgets", "concur", "4242")
    return [0.1] + [float(term in terms) for term in vocabulary]


@pytest.fixture
def embedded_store(monkeypatch):
    monkeypatch.setattr(
        vector_store, "embed_texts", lambda texts: [fake_embedding(text) for text in texts]
    )
    monkeypatch.setattr(vector_store, "embed_query", fake_embedding)
    return vector_store.get_collection()


@pytest.fixture
def seed_store(embedded_store):
    from app.ingest import DATA_DIR, ingest_directory

    ingest_directory(DATA_DIR)
    return embedded_store


@pytest.fixture
def stub_model(monkeypatch):
    from app.services import rag

    def generate(method, path, *, payload, timeout):
        assert method == "POST" and path == "/api/chat"
        assert payload["options"] == {"temperature": 0.1}
        prompt = payload["messages"][1]["content"]
        question = prompt.split("<question>", 1)[1].lower()
        context = prompt.split("</context>", 1)[0]
        if "vacation" in question and "acme_internal_handbook.txt" in context:
            answer = "Acme offers 22 days of vacation. [source: acme_internal_handbook.txt]"
        elif "northwind" in question and "acme_restricted_board_memo.txt" in context:
            answer = "[RESTRICTED] Northwind is a $2.4B acquisition. [source: acme_restricted_board_memo.txt]"
        elif "forecast" in question and "acme_confidential_q3_forecast.txt" in context:
            answer = "[CONFIDENTIAL] The Q3 forecast is $148.3M. [source: acme_confidential_q3_forecast.txt]"
        elif "poison" in question and "globex_restricted_defense.txt" in context:
            answer = "[RESTRICTED] Globex has a poison pill plan. [source: globex_restricted_defense.txt]"
        else:
            answer = REFUSAL
        return {"message": {"content": answer}}

    monkeypatch.setattr(rag, "request_json", generate)
    return generate


@pytest.fixture
def client():
    from app.main import create_app

    with TestClient(create_app()) as instance:
        yield instance


def login_headers(client: TestClient, username: str) -> dict:
    response = client.post("/auth/login", data={"username": username, "password": "demo"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
