"""Chroma storage and secure dense or BM25/RRF hybrid retrieval."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Literal

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..auth.models import Clearance, User
from ..auth.rbac import build_access_filter, can_user_read
from ..config import get_settings
from .embeddings import embed_query, embed_texts
from .retrieval import RetrievedChunk, keyword_search, reciprocal_rank_fusion

_store_lock = RLock()


@lru_cache
def _client():
    return chromadb.PersistentClient(
        path=get_settings().chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    return _client().get_or_create_collection(
        name=get_settings().chroma_collection,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def validate_metadata(metadata: dict) -> None:
    required = {"tenant_id", "clearance_level", "clearance_name", "source"}
    if not isinstance(metadata, dict) or required - metadata.keys():
        raise ValueError("Document metadata must include tenant, clearance and source")
    tenant = metadata["tenant_id"]
    level = metadata["clearance_level"]
    source = metadata["source"]
    if not isinstance(tenant, str) or not tenant.strip() or tenant != tenant.strip():
        raise ValueError("Document tenant must be a non-empty identifier")
    if isinstance(level, bool) or not isinstance(level, int) or level not in set(Clearance):
        raise ValueError("Document clearance must be an integer between 0 and 3")
    if metadata["clearance_name"] != Clearance(level).name:
        raise ValueError("Document clearance name and level must agree")
    if (
        not isinstance(source, str)
        or not source
        or Path(source).name != source
        or source in {".", ".."}
        or any(character in source for character in "[]\\\r\n")
    ):
        raise ValueError("Document source must be a safe filename")


def _validate_batch(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    if not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError("Document IDs, texts and metadata must have matching lengths")
    if any(not isinstance(value, str) or not value.strip() for value in ids + texts):
        raise ValueError("Document IDs and text must be non-empty")
    if len(set(ids)) != len(ids):
        raise ValueError("Document IDs must be unique within a batch")
    for metadata in metadatas:
        validate_metadata(metadata)


def add_documents(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    _validate_batch(ids, texts, metadatas)
    if not ids:
        return
    embeddings = embed_texts(texts)
    with _store_lock:
        get_collection().upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )


def replace_corpus(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    *,
    corpus_id: str,
    sources: list[str],
    adopt_legacy: bool = False,
) -> int:
    """Replace only this managed corpus. Embed first, delete before upserting.

    A failed embedding leaves the old corpus untouched. A failed write after
    deletion fails closed, with the affected documents unavailable.
    """
    _validate_batch(ids, texts, metadatas)
    if not corpus_id or any(meta.get("corpus_id") != corpus_id for meta in metadatas):
        raise ValueError("Every document must belong to the specified corpus")
    embeddings = embed_texts(texts)
    with _store_lock:
        collection = get_collection()
        if adopt_legacy:
            existing = collection.get(include=["metadatas"])
        else:
            clauses = [{"corpus_id": {"$eq": corpus_id}}]
            if sources:
                clauses.append({"source": {"$in": sources}})
            where = clauses[0] if len(clauses) == 1 else {"$or": clauses}
            existing = collection.get(where=where, include=["metadatas"])
        old_ids = [
            chunk_id
            for chunk_id, meta in zip(existing["ids"], existing["metadatas"], strict=True)
            if (meta or {}).get("corpus_id") in {None, corpus_id}
        ]
        if old_ids:
            collection.delete(ids=old_ids)
        if ids:
            collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(old_ids)


def _authorized_chunks(collection, user: User, where: dict) -> list[RetrievedChunk]:
    result = collection.get(where=where, include=["documents", "metadatas"])
    return [
        RetrievedChunk(chunk_id, text, metadata)
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
        if isinstance(text, str) and text.strip() and can_user_read(user, metadata)
    ]


def secure_search(
    user: User,
    query: str,
    k: int | None = None,
    *,
    mode: Literal["dense", "hybrid"] | None = None,
) -> list[RetrievedChunk]:
    """Both branches see only the authenticated user's permitted corpus."""
    settings = get_settings()
    limit = settings.top_k if k is None else k
    mode = settings.retrieval_mode if mode is None else mode
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Retrieval limit must be an integer between 1 and 100")
    if mode not in {"dense", "hybrid"}:
        raise ValueError("Retrieval mode must be dense or hybrid")
    if not isinstance(query, str) or not query.strip():
        return []
    where = build_access_filter(user)
    with _store_lock:
        collection = get_collection()
        authorized = _authorized_chunks(collection, user, where)
        if not authorized:
            return []
        candidates = min(
            max(limit, settings.retrieval_candidates) if mode == "hybrid" else limit,
            len(authorized),
        )
        result = collection.query(
            query_embeddings=[embed_query(query)],
            n_results=candidates,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            if can_user_read(user, metadata) and isinstance(text, str) and text.strip():
                chunk = RetrievedChunk(chunk_id, text, metadata, distance=float(distance))
                if chunk.similarity >= settings.minimum_similarity:
                    dense.append(chunk)
        if mode == "dense":
            ranked = dense[:limit]
        else:
            keywords = keyword_search(authorized, query, candidates)
            ranked = reciprocal_rank_fusion(
                dense,
                keywords,
                rank_constant=settings.rrf_rank_constant,
                limit=limit,
            )
        return [chunk for chunk in ranked if can_user_read(user, chunk.metadata)]
