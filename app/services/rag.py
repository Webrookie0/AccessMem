"""AccessMem question answering with authorized context and verified source IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from ..auth.models import User
from ..auth.rbac import build_access_filter
from ..config import get_settings
from . import guardrails
from .audit import log_event
from .ollama import OllamaError, request_json
from .retrieval import RetrievedChunk
from .vector_store import secure_search


@dataclass
class ChatAnswer:
    answer: str
    sources: list[dict]
    blocked: bool = False
    block_reason: str | None = None
    refused: bool = False
    retrieval_mode: str = "hybrid"
    # Used by the offline evaluator; never included in the API response.
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list, repr=False)


def _format_block(chunk: RetrievedChunk) -> str:
    return f"[source: {chunk.metadata['source']}]\n{chunk.text}"


def _sources_payload(chunks: list[RetrievedChunk], answer: str) -> list[dict]:
    cited_sources = {source.strip() for source in guardrails._CITATIONS.findall(answer)}
    return [
        {
            "id": chunk.id,
            "source": chunk.metadata["source"],
            "clearance": chunk.metadata["clearance_name"],
            "tenant_id": chunk.metadata["tenant_id"],
            "score": None if chunk.similarity is None else round(chunk.similarity, 4),
            "rrf_score": None if chunk.rrf_score is None else round(chunk.rrf_score, 6),
            "dense_rank": chunk.dense_rank,
            "keyword_rank": chunk.keyword_rank,
        }
        for chunk in chunks
        if chunk.metadata["source"] in cited_sources
    ]


def answer_question(
    user: User,
    question: str,
    *,
    mode: Literal["dense", "hybrid"] | None = None,
) -> ChatAnswer:
    settings = get_settings()
    mode = settings.retrieval_mode if mode is None else mode
    started = perf_counter()
    fields = {"user": user.username, "role": user.role, "tenant": user.tenant_id}
    checked = guardrails.check_input(question)
    if not checked.allowed:
        log_event("blocked_input", **fields, reason=checked.reason, question=question)
        return ChatAnswer(
            checked.reason or "Request blocked.",
            [],
            blocked=True,
            block_reason=checked.reason,
            retrieval_mode=mode,
        )
    try:
        chunks = secure_search(user, question, mode=mode)
    except OllamaError:
        log_event("model_error", **fields, stage="embedding")
        raise
    log_event(
        "retrieval",
        **fields,
        clearance=user.clearance.name,
        question=question,
        filters=build_access_filter(user),
        retrieval_mode=mode,
        retrieval_ms=round((perf_counter() - started) * 1000, 2),
        retrieved=[
            {
                "id": chunk.id,
                "source": chunk.metadata["source"],
                "clearance": chunk.metadata["clearance_name"],
            }
            for chunk in chunks
        ],
    )
    if not chunks:
        return ChatAnswer(guardrails.REFUSAL, [], refused=True, retrieval_mode=mode)
    messages = guardrails.build_prompt(question, [_format_block(chunk) for chunk in chunks])
    try:
        data = request_json(
            "POST",
            "/api/chat",
            payload={
                "model": settings.ollama_chat_model,
                "messages": messages,
                "options": {"temperature": 0.1},
                "stream": False,
            },
            timeout=settings.ollama_chat_timeout,
        )
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama returned an invalid chat response")
        raw = message["content"].strip()
        if not raw:
            raise OllamaError("Ollama returned an empty answer")
    except OllamaError:
        log_event("model_error", **fields, stage="generation")
        raise
    checked = guardrails.filter_output(raw, user)
    if checked.allowed:
        checked = guardrails.validate_citations(raw, {chunk.metadata["source"] for chunk in chunks})
    if not checked.allowed:
        log_event("blocked_output", **fields, reason=checked.reason)
        return ChatAnswer(
            checked.reason or "Response withheld.",
            [],
            blocked=True,
            block_reason=checked.reason,
            retrieval_mode=mode,
            retrieved_chunks=chunks,
        )
    refused = raw == guardrails.REFUSAL
    sources = [] if refused else _sources_payload(chunks, raw)
    log_event(
        "answer",
        **fields,
        refused=refused,
        source_count=len(sources),
        duration_ms=round((perf_counter() - started) * 1000, 2),
    )
    return ChatAnswer(
        raw,
        sources,
        refused=refused,
        retrieval_mode=mode,
        retrieved_chunks=chunks,
    )
