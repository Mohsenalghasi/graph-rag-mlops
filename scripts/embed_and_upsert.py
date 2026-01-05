from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Generator
from itertools import islice

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState,
)

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
load_dotenv(override=True)

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
SEARCH_KEY = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]

AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AOAI_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "preview")
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
EMBED_DIM = int(os.environ["AZURE_OPENAI_EMBEDDINGS_DIM"])

CURATED_DIR = Path("data/curated")
FAILED_DIR = Path("data/failed_batches")
FAILED_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 50

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_key(raw: str) -> str:
    """Azure Search–safe document key"""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_retryable_http(resp: requests.Response) -> bool:
    return resp.status_code in (408, 409, 429, 500, 502, 503, 504)


def should_retry(exc: Exception) -> bool:
    return isinstance(exc, requests.exceptions.RequestException)


def before_sleep(retry_state: RetryCallState):
    exc = retry_state.outcome.exception()
    print(
        f"[RETRY] {type(exc).__name__}: {exc}. attempt={retry_state.attempt_number}",
        flush=True,
    )


# ---------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception(should_retry),
    before_sleep=before_sleep,
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    url = f"{AOAI_ENDPOINT}/openai/v1/embeddings?api-version={AOAI_VERSION}"
    payload = {"model": EMBED_DEPLOYMENT, "input": texts}

    r = requests.post(
        url,
        headers={"api-key": AOAI_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if is_retryable_http(r):
        r.raise_for_status()

    r.raise_for_status()

    data = sorted(r.json()["data"], key=lambda x: x["index"])
    vecs = [row["embedding"] for row in data]

    if vecs and len(vecs[0]) != EMBED_DIM:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vecs[0])}, expected {EMBED_DIM}"
        )

    return vecs


# ---------------------------------------------------------------------
# AZURE SEARCH UPSERT
# ---------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(should_retry),
    before_sleep=before_sleep,
)
def upsert_docs(docs: List[Dict[str, Any]]) -> None:
    url = (
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index"
        f"?api-version=2024-07-01"
    )

    r = requests.post(
        url,
        headers={"api-key": SEARCH_KEY, "Content-Type": "application/json"},
        json={"value": docs},
        timeout=60,
    )

    if is_retryable_http(r):
        r.raise_for_status()

    r.raise_for_status()

    resp = r.json()
    failed = [x for x in resp.get("value", []) if x.get("status") is not True]

    if failed:
        raise requests.exceptions.RequestException(
            f"Partial upsert failure: {failed[:2]}"
        )


# ---------------------------------------------------------------------
# BATCHING
# ---------------------------------------------------------------------
def batch_jsonl(
    file_path: Path, batch_size: int
) -> Generator[List[Dict[str, Any]], None, None]:
    with file_path.open("r", encoding="utf-8") as f:
        it = (json.loads(line) for line in f if line.strip())
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    files = sorted(CURATED_DIR.glob("*.jsonl"))
    if not files:
        print("No curated .jsonl files found in data/curated")
        return

    total = 0

    for fp in files:
        print(f"FILE: {fp.name}", flush=True)

        for b, batch in enumerate(batch_jsonl(fp, BATCH_SIZE), start=1):
            try:
                texts = [x.get("content", "") for x in batch]
                vectors = embed_texts(texts)

                docs: List[Dict[str, Any]] = []

                for item, vec in zip(batch, vectors):
                    raw_id = str(item["id"])
                    safe_id = safe_key(raw_id)

                    source_path = item.get("source_path") or item.get("source") or ""
                    source_basename = (
                        Path(source_path).name
                        if source_path
                        else item.get("source", "")
                    )

                    doc = {
                        "@search.action": "mergeOrUpload",
                        "id": safe_id,
                        "content": item.get("content", ""),
                        "embedding": vec,
                        "source": item.get("source") or source_basename,
                        "source_original": source_path,
                        "source_basename": source_basename,
                        "doc_type": item.get("doc_type") or "manual",
                        "page": int(item.get("page_start") or 1),
                        "chunk_id": safe_id,
                    }

                    docs.append(doc)

                upsert_docs(docs)
                total += len(docs)
                print(
                    f"  batch {b}: upserted {len(docs)} (total={total})",
                    flush=True,
                )

            except Exception as e:
                out = FAILED_DIR / f"{fp.stem}.batch{b}.failed.jsonl"
                with out.open("w", encoding="utf-8") as w:
                    for row in batch:
                        w.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(
                    f"[FAIL] {fp.name} batch {b}: {type(e).__name__}: {e}",
                    flush=True,
                )

    print(f"DONE. Total indexed: {total}", flush=True)


if __name__ == "__main__":
    main()
