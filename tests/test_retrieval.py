import pytest

from app.auth.models import User
from app.auth.rbac import can_user_read
from app.services import vector_store
from app.services.retrieval import RetrievedChunk, keyword_search, reciprocal_rank_fusion


def chunk(identifier, text="text", **metadata):
    return RetrievedChunk(identifier, text, metadata, distance=0.2)


def test_rrf_combines_independent_rankings_and_keeps_similarity():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    result = reciprocal_rank_fusion([a, b], [b, c])
    assert [item.id for item in result] == ["b", "a", "c"]
    assert result[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[0].similarity == pytest.approx(0.8)
    assert result[0].dense_rank == 2 and result[0].keyword_rank == 1


def test_rrf_deduplicates_a_branch_and_breaks_ties_by_id():
    a, b = chunk("a"), chunk("b")
    result = reciprocal_rank_fusion([b, b], [a])
    assert [item.id for item in result] == ["a", "b"]
    assert result[1].rrf_score == pytest.approx(1 / 61)


def test_keyword_search_handles_single_document_and_exact_project_name():
    expected = chunk("one", "Budget data", title="Project Northwind")
    assert keyword_search([expected], "NORTHWIND", 3) == [expected]
    assert keyword_search([expected], "unrelated query", 3) == []


def test_keyword_search_does_not_return_baseline_only_matches():
    a, b = chunk("a", "vacation policy"), chunk("b", "board acquisition")
    assert [item.id for item in keyword_search([a, b], "vacation", 4)] == ["a"]


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
@pytest.mark.parametrize(
    "role,tenant",
    [
        ("junior", "acme"),
        ("manager", "acme"),
        ("csuite", "acme"),
        ("csuite", "globex"),
    ],
)
def test_real_chroma_retrieval_never_crosses_permissions(seed_store, mode, role, tenant):
    user = User(username="test", role=role, tenant_id=tenant)
    result = vector_store.secure_search(user, "Northwind forecast poison vacation", mode=mode)
    assert all(can_user_read(user, item.metadata) for item in result)
    if role == "junior":
        assert not any("restricted" in item.metadata["source"] for item in result)
    if tenant == "globex":
        assert all(item.metadata["source"].startswith("globex") for item in result)


def test_bm25_corpus_is_authorized_before_ranking(seed_store, monkeypatch):
    user = User(username="alice", role="junior", tenant_id="acme")
    original = vector_store.keyword_search
    observed = []

    def inspect_corpus(chunks, query, limit):
        observed.extend(chunks)
        assert all(can_user_read(user, item.metadata) for item in chunks)
        return original(chunks, query, limit)

    monkeypatch.setattr(vector_store, "keyword_search", inspect_corpus)
    vector_store.secure_search(user, "Northwind", mode="hybrid")
    assert observed and len(observed) == 2


def test_hybrid_recovers_keyword_only_match_below_dense_threshold(embedded_store):
    vector_store.add_documents(
        ["exact", "other"],
        ["A unique ZX493 contract", "Other information"],
        [
            {
                "tenant_id": "acme",
                "clearance_level": 1,
                "clearance_name": "INTERNAL",
                "source": "contract.md",
            },
            {
                "tenant_id": "acme",
                "clearance_level": 1,
                "clearance_name": "INTERNAL",
                "source": "other.md",
            },
        ],
    )
    from app.config import get_settings

    get_settings().minimum_similarity = 1.0
    monkey_user = User(username="alice", role="junior", tenant_id="acme")
    # Set a deliberately different query embedding so dense candidates are filtered.
    from unittest.mock import patch

    with patch.object(vector_store, "embed_query", return_value=[0.0, 1.0, 0, 0, 0, 0, 0, 0]):
        assert vector_store.secure_search(monkey_user, "ZX493", mode="dense") == []
        result = vector_store.secure_search(monkey_user, "ZX493", mode="hybrid")
    assert [item.id for item in result] == ["exact"]
    assert result[0].distance is None and result[0].rrf_score is not None


def test_empty_authorized_corpus_does_not_call_embedding(embedded_store, monkeypatch):
    def fail(_):
        raise AssertionError("No model request should be made")

    monkeypatch.setattr(vector_store, "embed_query", fail)
    user = User(username="nobody", role="csuite", tenant_id="other")
    assert vector_store.secure_search(user, "Northwind") == []


@pytest.mark.parametrize("limit", [0, -1, 101, True])
def test_invalid_retrieval_limit_is_rejected(limit):
    user = User(username="alice", role="junior", tenant_id="acme")
    with pytest.raises(ValueError):
        vector_store.secure_search(user, "query", k=limit)
