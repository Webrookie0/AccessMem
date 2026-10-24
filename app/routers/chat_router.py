from typing import Literal

from chromadb.errors import ChromaError
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..auth.jwt import get_current_user
from ..auth.models import User
from ..rate_limit import limit_chat
from ..services.audit import log_event
from ..services.ollama import OllamaError
from ..services.rag import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    retrieval_mode: Literal["dense", "hybrid"] | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class ChatSource(BaseModel):
    id: str
    source: str
    clearance: str
    tenant_id: str
    score: float | None = None
    rrf_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    blocked: bool = False
    block_reason: str | None = None
    refused: bool = False
    retrieval_mode: Literal["dense", "hybrid"]


@router.post("", response_model=ChatResponse, dependencies=[Depends(limit_chat)])
def chat(req: ChatRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    try:
        result = answer_question(user, req.question, mode=req.retrieval_mode)
    except OllamaError as exc:
        raise HTTPException(
            503, "Model service unavailable. Start Ollama and pull the configured models."
        ) from exc
    except ChromaError as exc:
        log_event(
            "retrieval_error",
            user=user.username,
            tenant=user.tenant_id,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            503, "Document index unavailable. Check the index and re-run ingestion."
        ) from exc
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        blocked=result.blocked,
        block_reason=result.block_reason,
        refused=result.refused,
        retrieval_mode=result.retrieval_mode,
    )
