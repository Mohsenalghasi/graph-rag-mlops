# core/embedder.py
import os
from functools import lru_cache
from typing import List

from openai import AzureOpenAI


def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is missing in .env")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is missing in .env")
    if not api_version or api_version.lower() == "preview":
        raise RuntimeError("AZURE_OPENAI_API_VERSION must be a real version like 2024-02-15-preview")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@lru_cache(maxsize=2048)
def embed_query(text: str) -> List[float]:
    """
    Azure OpenAI embeddings: text -> vector
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Query text is empty.")

    deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT is missing in .env")

    client = _get_client()
    resp = client.embeddings.create(model=deployment, input=text)
    vec = resp.data[0].embedding

    if not isinstance(vec, list) or len(vec) == 0:
        raise RuntimeError("Embedding response is empty/unexpected.")
    return vec
