import os
from fastapi import FastAPI
from pydantic import BaseModel

from app.logging_conf import configure_logging

configure_logging()

app = FastAPI(title="Graph-RAG MLOps", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    env: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", env=os.getenv("APP_ENV", "local"))
