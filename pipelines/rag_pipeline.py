# pipelines/rag_pipeline.py
from core.embedder import embed_query
from core.search import vector_search
from core.rerank import rerank
from core.prompt import build_context, build_messages
from core.generate import generate_answer

def run(question: str, top_k: int = 20, top_n: int = 5) -> str:
    vec = embed_query(question)

    # retrieve more
    docs = vector_search(vec, top_k=top_k)

    # rerank to fewer
    docs = rerank(question, docs, top_n=top_n)

    context = build_context(docs)
    messages = build_messages(question, context)
    return generate_answer(messages)
