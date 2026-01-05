import json
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv(override=True)

from pipelines.rag_pipeline import run  # uses top_k/top_n we pass below


EVAL_PATH = Path("data/eval/questions.jsonl")
OUT_PATH = Path("data/eval/report.json")


def has_citation(answer: str) -> bool:
    # our format is: [<source> | page X]
    return "[" in answer and "| page" in answer and "]" in answer


def keywords_present(answer: str, must: List[str]) -> List[str]:
    a = (answer or "").lower()
    missing = []
    for k in must:
        if k.lower() not in a:
            missing.append(k)
    return missing


def main(top_k: int = 20, top_n: int = 5):
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PATH}. Create it first.")

    rows = []
    lines = EVAL_PATH.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        item = json.loads(line)
        q = item["q"]
        must = item.get("must_contain", [])

        t0 = time.perf_counter()
        ans = run(q, top_k=top_k, top_n=top_n)
        ms = round((time.perf_counter() - t0) * 1000, 2)

        missing = keywords_present(ans, must)
        cit_ok = has_citation(ans)

        rows.append({
            "i": i,
            "q": q,
            "latency_ms": ms,
            "citations_ok": cit_ok,
            "missing_keywords": missing,
            "answer": ans,
        })

        print(f"[{i}/{len(lines)}] {ms} ms | citations={cit_ok} | missing={len(missing)}")

    # summary
    n = len(rows)
    avg_ms = round(sum(r["latency_ms"] for r in rows) / max(n, 1), 2)
    cit_rate = round(sum(1 for r in rows if r["citations_ok"]) / max(n, 1), 3)
    kw_pass = round(sum(1 for r in rows if len(r["missing_keywords"]) == 0) / max(n, 1), 3)

    report = {
        "config": {"top_k": top_k, "top_n": top_n},
        "summary": {
            "n": n,
            "avg_latency_ms": avg_ms,
            "citation_pass_rate": cit_rate,
            "keyword_pass_rate": kw_pass,
        },
        "rows": rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", OUT_PATH)
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main(top_k=30, top_n=8)
