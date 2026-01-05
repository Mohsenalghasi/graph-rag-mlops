# core/search.py
import os
from typing import Any, Dict, List
import requests


def vector_search(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    ep = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
    key = os.environ["AZURE_SEARCH_API_KEY"]
    idx = os.environ["AZURE_SEARCH_INDEX_NAME"]

    url = f"{ep}/indexes/{idx}/docs/search?api-version=2024-07-01"

    payload = {
        "vectorQueries": [
            {"kind": "vector", "vector": query_vector, "fields": "embedding", "k": top_k}
        ],
        "top": top_k,
        "select": "id,content,source_original,page,chunk_id",
    }

    r = requests.post(
        url,
        headers={"api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Search error {r.status_code}: {r.text[:800]}")

    return r.json().get("value", [])
