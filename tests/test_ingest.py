from pathlib import Path

import pytest

from app.auth.models import User
from app.ingest import _chunk, _parse_front_matter, ingest_directory
from app.services import vector_store
from app.services.ollama import OllamaError


def document(path: Path, text: str, clearance: str = "INTERNAL"):
    path.write_text(f"tenant: acme\nclearance: {clearance}\ntitle: Test\n---\n{text}")


def test_chunking_preserves_markdown_and_terminates():
    text = "# Vacation policy\n\n- First point\n- Second point\n\n" + "many words " * 50
    chunks = _chunk(text, size=100, overlap=20)
    assert chunks[0].startswith("# Vacation policy\n\n- First point")
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    assert chunks[-1].endswith("many words")


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunk_settings_fail(size, overlap):
    with pytest.raises(ValueError):
        _chunk("body", size=size, overlap=overlap)


def test_front_matter_handles_crlf_and_rejects_duplicate_permissions():
    metadata, body = _parse_front_matter("tenant: acme\r\nclearance: INTERNAL\r\n---\r\nBody")
    assert metadata["tenant"] == "acme" and body == "Body"
    with pytest.raises(ValueError):
        _parse_front_matter("tenant: acme\ntenant: globex\n---\nBody")


@pytest.mark.parametrize("extension", [".md", ".txt"])
def test_synchronize_removes_stale_chunks_and_deleted_documents(
    embedded_store, tmp_path, extension
):
    first = tmp_path / f"first{extension}"
    second = tmp_path / f"second{extension}"
    document(first, "Vacation " * 300)
    document(second, "Other information")
    initial = ingest_directory(tmp_path)
    assert initial["chunks"] > 2
    document(first, "Short vacation policy")
    second.unlink()
    result = ingest_directory(tmp_path)
    assert result["chunks"] == 1 and embedded_store.count() == 1
    assert embedded_store.get()["ids"] == ["seed:acme:first::chunk0"]


def test_changed_clearance_removes_old_accessible_chunks(embedded_store, tmp_path):
    path = tmp_path / "policy.md"
    document(path, "Vacation details")
    ingest_directory(tmp_path)
    alice = User(username="alice", role="junior", tenant_id="acme")
    assert vector_store.secure_search(alice, "Vacation")
    document(path, "Vacation details", clearance="RESTRICTED")
    ingest_directory(tmp_path)
    assert vector_store.secure_search(alice, "Vacation") == []


def test_other_corpus_survives_synchronization(embedded_store, tmp_path):
    vector_store.add_documents(
        ["other-corpus"],
        ["Vacation notes"],
        [
            {
                "tenant_id": "acme",
                "clearance_level": 1,
                "clearance_name": "INTERNAL",
                "source": "other.md",
                "corpus_id": "other",
            }
        ],
    )
    ingest_directory(tmp_path)
    assert embedded_store.get()["ids"] == ["other-corpus"]


def test_failed_embedding_leaves_old_corpus_untouched(embedded_store, tmp_path, monkeypatch):
    path = tmp_path / "policy.md"
    document(path, "Vacation policy")
    ingest_directory(tmp_path)
    existing = embedded_store.get()["ids"]
    document(path, "Updated vacation policy")

    def fail(_):
        raise OllamaError("Unavailable")

    monkeypatch.setattr(vector_store, "embed_texts", fail)
    with pytest.raises(OllamaError):
        ingest_directory(tmp_path)
    assert embedded_store.get()["ids"] == existing


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"tenant_id": "", "clearance_level": 1, "clearance_name": "INTERNAL", "source": "a.md"},
        {
            "tenant_id": "acme",
            "clearance_level": True,
            "clearance_name": "INTERNAL",
            "source": "a.md",
        },
        {
            "tenant_id": "acme",
            "clearance_level": 1,
            "clearance_name": "RESTRICTED",
            "source": "a.md",
        },
        {
            "tenant_id": "acme",
            "clearance_level": 1,
            "clearance_name": "INTERNAL",
            "source": "../a.md",
        },
    ],
)
def test_invalid_metadata_is_rejected_before_embedding(metadata, monkeypatch):
    def fail(_):
        raise AssertionError("Invalid metadata must fail before embedding")

    monkeypatch.setattr(vector_store, "embed_texts", fail)
    with pytest.raises(ValueError):
        vector_store.add_documents(["a"], ["body"], [metadata])
