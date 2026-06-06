from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth_router, chat_router

app = FastAPI(
    title="Secure RAG with RBAC",
    description="Document assistant with role-based access control at the vector store layer.",
    version="1.0.0",
)

# CORS — locked down to localhost dev origins. Tighten for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(chat_router.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
