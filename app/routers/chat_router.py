from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth.jwt import get_current_user
from ..auth.models import User
from ..services.rag import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    blocked: bool = False
    block_reason: str | None = None


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    result = answer_question(user, req.question)
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        blocked=result.blocked,
        block_reason=result.block_reason,
    )
