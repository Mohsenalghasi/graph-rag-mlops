# core/search.py
import os
from typing import Any, Dict, List

import requests


def search_top_k(query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """
    Pure vector search in Azure AI Search.
    Returns list of docs with fields: source_basename, page, content, chunk_id.
    """
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "")
    if not endpoint or not api_key or not index_name:
        raise RuntimeError(
            "Missing Azure Search env vars: AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_API_KEY / AZURE_SEARCH_INDEX_NAME"
        )

    url = f"{endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01"

    body = {
        "search": "",
        "top": int(k),
        "select": "chunk_id,source,source_basename,page,content,doc_type",
        "vectorQueries": [
            {"kind": "vector", "vector": query_vector, "fields": "embedding", "k": int(k)}
        ],
    }

    r = requests.post(
        url,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Azure Search failed: {r.status_code} {r.text[:300]}")

    hits = r.json().get("value", [])
    if not isinstance(hits, list):
        return []
    return hits
