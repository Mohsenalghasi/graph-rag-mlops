import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.logging_conf import configure_logging
from core.generate import answer_question

configure_logging()

app = FastAPI(
    title="Graph-RAG MLOps",
    version="1.0.0",
    description="Production-grade retrieval-first RAG API with evaluation gating",
)


# -----------------------------
# Schemas
# -----------------------------
class HealthResponse(BaseModel):
    status: str
    env: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    k: int = Field(5, ge=1, le=20, description="Top-K retrieval size")


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    k: int
    error: str | None = None


# -----------------------------
# Routes
# -----------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", env=os.getenv("APP_ENV", "local"))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer_question(query=req.query, k=req.k)
        # result is expected: {"answer":..., "citations":..., "k":..., "error": optional}
        return AskResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            k=result.get("k", req.k),
            error=result.get("error"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error while processing the request")
