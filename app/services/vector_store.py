"""Chroma vector-store wrapper. All queries MUST go through `secure_search`,
which forces a server-side metadata filter built from the authenticated user.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..auth.models import User
from ..auth.rbac import build_access_filter, can_user_read
from ..config import get_settings
from .embeddings import embed_query, embed_texts


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    distance: float


@lru_cache
def _client():
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection():
    settings = get_settings()
    return _client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    """Index a batch. `metadatas` MUST contain `tenant_id`, `clearance_level`,
    `clearance_name`, and `source`."""
    required = {"tenant_id", "clearance_level", "clearance_name", "source"}
    for m in metadatas:
        missing = required - m.keys()
        if missing:
            raise ValueError(f"document metadata missing required keys: {missing}")

    embeddings = embed_texts(texts)
    get_collection().upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def secure_search(user: User, query: str, k: int | None = None) -> list[RetrievedChunk]:
    """Retrieval entrypoint enforcing RBAC. The `where` filter is derived
    SOLELY from the authenticated User — never from caller input."""
    settings = get_settings()
    k = k or settings.top_k

    where = build_access_filter(user)
    q_emb = embed_query(query)

    res = get_collection().query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        # Defence in depth: re-verify each returned doc is permitted.
        if not can_user_read(user, meta or {}):
            continue
        chunks.append(RetrievedChunk(id=i, text=doc, metadata=meta or {}, distance=float(dist)))
    return chunks
