"""End-to-end RAG pipeline tying together RBAC retrieval, guardrails, and the LLM."""
from __future__ import annotations

from dataclasses import dataclass
import requests
from functools import lru_cache

from ..auth.models import User
from ..config import get_settings
from . import guardrails
from .audit import log_event
from .vector_store import RetrievedChunk, secure_search


@dataclass
class ChatAnswer:
    answer: str
    sources: list[dict]
    blocked: bool = False
    block_reason: str | None = None


def _get_base_url() -> str:
    return get_settings().ollama_base_url.rstrip("/")


def _format_block(chunk: RetrievedChunk) -> str:
    src = chunk.metadata.get("source", "unknown")
    return f"[source: {src}]\n{chunk.text}"


def _sources_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "id": c.id,
            "source": c.metadata.get("source"),
            "clearance": c.metadata.get("clearance_name"),
            "tenant_id": c.metadata.get("tenant_id"),
            "score": round(1.0 - c.distance, 4),
        }
        for c in chunks
    ]


def answer_question(user: User, question: str) -> ChatAnswer:
    # 1. Input guardrail
    in_check = guardrails.check_input(question)
    if not in_check.allowed:
        log_event(
            "blocked_input",
            user=user.username, role=user.role, tenant=user.tenant_id,
            reason=in_check.reason, question=question,
        )
        return ChatAnswer(
            answer=in_check.reason or "Request blocked.",
            sources=[], blocked=True, block_reason=in_check.reason,
        )

    # 2. RBAC-scoped retrieval
    chunks = secure_search(user, question)
    log_event(
        "retrieval",
        user=user.username, role=user.role, tenant=user.tenant_id,
        clearance=user.clearance.name,
        question=question,
        retrieved=[{"id": c.id, "source": c.metadata.get("source"),
                    "clearance": c.metadata.get("clearance_name")} for c in chunks],
    )

    if not chunks:
        return ChatAnswer(
            answer="I don't have access to information that answers that question.",
            sources=[],
        )

    # 3. LLM call with hardened system prompt using Ollama
    messages = guardrails.build_prompt(question, [_format_block(c) for c in chunks])
    settings = get_settings()
    
    try:
        base_url = _get_base_url()
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": settings.ollama_chat_model,
                "messages": messages,
                "temperature": 0.1,
                "stream": False,
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        raw = (data.get("message", {}).get("content", "") or "").strip()
    except requests.exceptions.RequestException as e:
        log_event(
            "llm_error",
            user=user.username, role=user.role, tenant=user.tenant_id,
            error=str(e),
        )
        return ChatAnswer(
            answer="Failed to generate response. Please try again.",
            sources=[], blocked=True, block_reason=str(e),
        )

    # 4. Output guardrail
    out_check = guardrails.filter_output(raw, user)
    if not out_check.allowed:
        log_event(
            "blocked_output",
            user=user.username, role=user.role, tenant=user.tenant_id,
            reason=out_check.reason,
        )
        return ChatAnswer(
            answer=out_check.reason or "Response withheld.",
            sources=[], blocked=True, block_reason=out_check.reason,
        )

    return ChatAnswer(answer=out_check.sanitized_text or raw, sources=_sources_payload(chunks))
