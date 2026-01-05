# core/rerank.py
from typing import Any, Dict, List, Tuple
from sentence_transformers import CrossEncoder

_ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    if not docs:
        return []

    pairs: List[Tuple[str, str]] = [(query, (d.get("content") or "")) for d in docs]
    scores = _ce.predict(pairs)

    for d, s in zip(docs, scores):
        d["_rerank_score"] = float(s)

    return sorted(docs, key=lambda x: x["_rerank_score"], reverse=True)[:top_n]
