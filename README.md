# Graph-RAG MLOps (End-to-End)

This repo will evolve into a full project:
- Ingestion (landing → staging → curated)
- Embeddings + Vector store (FAISS local / Azure Search)
- Graph RAG (LangGraph)
- Evaluation + Drift gates
- MLflow tracking
- CI/CD + Docker deployment

## Run locally (without Docker)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install .
uvicorn app.main:app --reload
