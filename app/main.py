"""AccessMem API application."""

import re
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from chromadb.errors import ChromaError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth.users import initialize_user_store
from .config import get_settings
from .routers import auth_router, chat_router
from .services.audit import log_event, request_id
from .services.ollama import OllamaError, available_models
from .services.vector_store import get_collection

VERSION = "1.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_user_store()
    get_collection()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="AccessMem",
        version=VERSION,
        lifespan=lifespan,
        description="Document question answering with tenant-scoped RBAC and BM25/RRF hybrid retrieval.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    application.include_router(auth_router.router)
    application.include_router(chat_router.router)

    @application.middleware("http")
    async def trace_request(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        identifier = incoming if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", incoming) else str(uuid4())
        token = request_id.set(identifier)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = identifier
            return response
        finally:
            log_event(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
            request_id.reset(token)

    @application.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "project": "AccessMem", "version": VERSION}

    @application.get("/ready", tags=["meta"])
    def ready():
        try:
            models = available_models()
            expected = {settings.ollama_chat_model, settings.ollama_embed_model}
            missing = [
                model
                for model in expected
                if model not in models and (model + ":latest") not in models
            ]
            if missing:
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "missing_models": sorted(missing)},
                )
            if get_collection().count() == 0:
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "detail": "Run document ingestion"},
                )
        except OllamaError:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": "Ollama unavailable"},
            )
        except ChromaError:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": "Document index unavailable"},
            )
        return {"status": "ready"}

    return application


app = create_app()
