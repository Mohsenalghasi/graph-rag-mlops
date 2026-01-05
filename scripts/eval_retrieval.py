# scripts/eval_retrieval.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
from dotenv import load_dotenv


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class EvalConfig:
    # Eval input
    eval_dir: Path
    eval_file: Path  # queries.jsonl

    # Retrieval
    k: int = 20
    timeout_sec: int = 60

    # Page alignment (truth vs indexed page numbering)
    # If your truth is "pdf|4" but your index stores 0-based pages,
    # set EVAL_PAGE_OFFSET=1 to subtract 1 from truth or add 1 to hits.
    page_offset: int = 0  # default: assume both are aligned

    # Outputs
    report_path: Path = Path("data/eval/report.json")
    details_path: Path = Path("data/eval/retrieval_details.json")


# -----------------------------
# Small helpers
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return str(v).strip()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_truth_key(s: str, page_offset: int = 0) -> str:
    """
    expected: "filename.pdf|12"
    apply optional offset to page number to align truth with index.
    """
    s = str(s).strip()
    if not s or "|" not in s:
        return s
    src, page = s.rsplit("|", 1)
    src = Path(src.strip()).name  # ensure basename
    try:
        page_i = int(page.strip())
        page_i = page_i + page_offset
    except Exception:
        return f"{src}|{page.strip()}"
    return f"{src}|{page_i}"


def hit_to_truth_key(hit: dict, page_offset: int = 0) -> str:
    """
    matches queries.jsonl format: source_basename|page
    """
    src = (hit.get("source_basename") or hit.get("source") or "").strip()
    if not src:
        return ""

    src = Path(src).name  # ensure basename

    page = hit.get("page", None)
    try:
        page_i = int(page)
        page_i = page_i + page_offset
    except Exception:
        return ""

    return f"{src}|{page_i}"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# HTTP session (connection pooling)
# -----------------------------
_SESSION = requests.Session()


# -----------------------------
# Azure OpenAI embeddings (endpoint version that WORKED for you)
# -----------------------------
def embed_query(query: str, timeout_sec: int) -> List[float]:
    """
    POST {AZURE_OPENAI_ENDPOINT}/openai/v1/embeddings?api-version=preview
    headers: api-key
    body: {"model": "<DEPLOYMENT_NAME>", "input": "..."}
    """
    ep = _env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    key = _env("AZURE_OPENAI_API_KEY")
    model = _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    url = f"{ep}/openai/v1/embeddings?api-version=preview"
    payload = {"model": model, "input": query}

    r = _SESSION.post(
        url,
        headers={"api-key": key, "Content-Type": "application/json"},
        json=payload,
        timeout=timeout_sec,
    )
    if r.status_code != 200:
        raise RuntimeError(f"AOAI embeddings failed: {r.status_code} {r.text[:300]}")

    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("AOAI embeddings returned empty data[]")

    vec = data[0].get("embedding", None)
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("AOAI embeddings missing embedding vector")

    return vec


# -----------------------------
# Azure AI Search vector query
# -----------------------------
def search_vector(query_vec: List[float], top_k: int, timeout_sec: int) -> List[Dict[str, Any]]:
    """
    Vector search:
    {AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2024-07-01
    vector field: embedding
    """
    endpoint = _env("AZURE_SEARCH_ENDPOINT").rstrip("/")
    key = _env("AZURE_SEARCH_API_KEY")
    index = _env("AZURE_SEARCH_INDEX_NAME")

    url = f"{endpoint}/indexes/{index}/docs/search?api-version=2024-07-01"

    body = {
        "search": "",  # pure vector
        "top": int(top_k),
        "select": "id,chunk_id,source,source_basename,doc_type,page,content",
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vec,
                "fields": "embedding",
                "k": int(top_k),
            }
        ],
    }

    r = _SESSION.post(
        url,
        headers={"api-key": key, "Content-Type": "application/json"},
        json=body,
        timeout=timeout_sec,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Search failed: {r.status_code} {r.text[:300]}")

    js = r.json()
    hits = js.get("value", [])
    if not isinstance(hits, list):
        return []
    return hits


# -----------------------------
# Eval file IO
# -----------------------------
def load_eval_rows(eval_path: Path) -> List[Dict[str, Any]]:
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Missing eval file: {eval_path}. Create it first at data/eval/queries.jsonl"
        )

    rows: List[Dict[str, Any]] = []
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "q" not in obj or "relevant" not in obj:
            raise ValueError("Each line must have keys: {'q':..., 'relevant':[...]}")

        rows.append({"q": str(obj["q"]), "relevant": list(obj["relevant"])})
    return rows


# -----------------------------
# MLflow (optional, never crash)
# -----------------------------
def try_log_mlflow(metrics: Dict[str, float], params: Dict[str, Any], artifacts: List[Path], run_name: str) -> None:
    try:
        import mlflow  # type: ignore
    except Exception:
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            return

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "graph_rag_mlops")

    try:
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

            for p in artifacts:
                try:
                    if p.exists():
                        mlflow.log_artifact(str(p))
                except Exception:
                    pass
    except Exception:
        return


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(hit_keys: List[str], truth_keys: List[str]) -> Dict[str, float]:
    """
    Computes:
      - hit@k: 1 if any relevant retrieved else 0
      - recall@k: fraction of relevant retrieved (true IR recall)
      - mrr@k: reciprocal rank of first relevant
    """
    truth_set = set(truth_keys)
    hit_set = set(hit_keys)

    hit_at_k = 1.0 if any(k in truth_set for k in hit_keys) else 0.0

    denom = max(1, len(truth_set))
    recall_at_k = len(hit_set.intersection(truth_set)) / float(denom)

    mrr_at_k = 0.0
    for rank, k in enumerate(hit_keys, start=1):
        if k in truth_set:
            mrr_at_k = 1.0 / rank
            break

    return {"hit@k": hit_at_k, "recall@k": recall_at_k, "mrr@k": mrr_at_k}


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    load_dotenv(override=True)

    eval_dir = Path("data/eval")
    eval_path = eval_dir / "queries.jsonl"

    # Backward compatibility: if user has questions.jsonl, use it
    if not eval_path.exists() and (eval_dir / "questions.jsonl").exists():
        eval_path = eval_dir / "questions.jsonl"

    cfg = EvalConfig(
        eval_dir=eval_dir,
        eval_file=eval_path,
        k=int(os.getenv("EVAL_K", "20")),
        timeout_sec=int(os.getenv("EVAL_TIMEOUT_SEC", "60")),
        page_offset=int(os.getenv("EVAL_PAGE_OFFSET", "0")),
        report_path=eval_dir / "report.json",
        details_path=eval_dir / "retrieval_details.json",
    )

    # sanity: ensure required env vars exist early
    _ = _env("AZURE_SEARCH_ENDPOINT")
    _ = _env("AZURE_SEARCH_API_KEY")
    _ = _env("AZURE_SEARCH_INDEX_NAME")
    _ = _env("AZURE_OPENAI_ENDPOINT")
    _ = _env("AZURE_OPENAI_API_KEY")
    _ = _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    rows = load_eval_rows(cfg.eval_file)

    details: List[Dict[str, Any]] = []

    hit_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    total_ms_sum = 0.0

    for i, row in enumerate(rows, start=1):
        q = row["q"]
        truth = [
            normalize_truth_key(x, page_offset=cfg.page_offset)
            for x in row["relevant"]
            if str(x).strip()
        ]

        t0 = time.perf_counter()
        embed_ms = 0.0
        search_ms = 0.0

        try:
            t_embed0 = time.perf_counter()
            qvec = embed_query(q, timeout_sec=cfg.timeout_sec)
            embed_ms = (time.perf_counter() - t_embed0) * 1000.0

            t_search0 = time.perf_counter()
            hits = search_vector(qvec, top_k=cfg.k, timeout_sec=cfg.timeout_sec)
            search_ms = (time.perf_counter() - t_search0) * 1000.0

            hit_keys = [hit_to_truth_key(h, page_offset=cfg.page_offset) for h in hits]
            hit_keys = [k for k in hit_keys if k]
            hit_keys = dedupe_preserve_order(hit_keys)  # avoid duplicate ranks

            m = compute_metrics(hit_keys, truth)
            hit_at_k, recall_at_k, mrr_at_k = m["hit@k"], m["recall@k"], m["mrr@k"]

        except Exception as e:
            hits = []
            hit_keys = []
            hit_at_k = 0.0
            recall_at_k = 0.0
            mrr_at_k = 0.0
            total_ms = (time.perf_counter() - t0) * 1000.0
            total_ms_sum += total_ms

            details.append(
                {
                    "query": q,
                    "relevant": truth,
                    "hit_keys": hit_keys,
                    "hits": hits,
                    f"hit@{cfg.k}": hit_at_k,
                    f"recall@{cfg.k}": recall_at_k,
                    f"mrr@{cfg.k}": mrr_at_k,
                    "latency_ms": round(total_ms, 2),
                    "embed_ms": round(embed_ms, 2),
                    "search_ms": round(search_ms, 2),
                    "error": str(e),
                }
            )

            hit_sum += hit_at_k
            recall_sum += recall_at_k
            mrr_sum += mrr_at_k

            print(
                f"[{i}/{len(rows)}] hit@{cfg.k}={hit_at_k:.3f} "
                f"recall@{cfg.k}={recall_at_k:.3f} mrr@{cfg.k}={mrr_at_k:.3f} "
                f"latency_ms={total_ms:.1f} ERROR={e}"
            )
            continue

        total_ms = (time.perf_counter() - t0) * 1000.0
        total_ms_sum += total_ms

        details.append(
            {
                "query": q,
                "relevant": truth,
                "hit_keys": hit_keys,
                "hits": hits[: cfg.k],
                f"hit@{cfg.k}": hit_at_k,
                f"recall@{cfg.k}": recall_at_k,
                f"mrr@{cfg.k}": mrr_at_k,
                "latency_ms": round(total_ms, 2),
                "embed_ms": round(embed_ms, 2),
                "search_ms": round(search_ms, 2),
            }
        )

        hit_sum += hit_at_k
        recall_sum += recall_at_k
        mrr_sum += mrr_at_k

        print(
            f"[{i}/{len(rows)}] hit@{cfg.k}={hit_at_k:.3f} "
            f"recall@{cfg.k}={recall_at_k:.3f} mrr@{cfg.k}={mrr_at_k:.3f} "
            f"latency_ms={total_ms:.1f}"
        )

    n = max(1, len(rows))
    report = {
        "ts_utc": now_iso(),
        "n": len(rows),
        "k": cfg.k,
        f"Hit@{cfg.k}": hit_sum / n,
        f"Recall@{cfg.k}": recall_sum / n,
        f"MRR@{cfg.k}": mrr_sum / n,
        "avg_latency_ms": total_ms_sum / n,
        "page_offset_applied": cfg.page_offset,
        "eval_file": str(cfg.eval_file).replace("\\", "/"),
    }

    cfg.eval_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg.details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    print(f"\nHit@{cfg.k}:    {report[f'Hit@{cfg.k}']:.3f}")
    print(f"Recall@{cfg.k}: {report[f'Recall@{cfg.k}']:.3f}")
    print(f"MRR@{cfg.k}:    {report[f'MRR@{cfg.k}']:.3f}")
    print(f"Avg latency:    {report['avg_latency_ms']:.1f} ms")
    print(f"Saved: {cfg.report_path}")
    print(f"Saved: {cfg.details_path}")

    # Optional MLflow logging
    try_log_mlflow(
        metrics={
            f"hit@{cfg.k}": report[f"Hit@{cfg.k}"],
            f"recall@{cfg.k}": report[f"Recall@{cfg.k}"],
            f"mrr@{cfg.k}": report[f"MRR@{cfg.k}"],
            "avg_latency_ms": report["avg_latency_ms"],
        },
        params={
            "k": cfg.k,
            "eval_file": str(cfg.eval_file),
            "page_offset_applied": cfg.page_offset,
        },
        artifacts=[cfg.report_path, cfg.details_path],
        run_name=f"eval_retrieval_k{cfg.k}",
    )


if __name__ == "__main__":
    main()
