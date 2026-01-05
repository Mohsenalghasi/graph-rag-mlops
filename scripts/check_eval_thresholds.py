import json
import os
from pathlib import Path

REPORT = Path("data/eval/report.json")

def main():
    if not REPORT.exists():
        raise SystemExit(f"Missing {REPORT}. Did eval run?")

    rep = json.loads(REPORT.read_text(encoding="utf-8"))

    k = rep.get("k")
    if not k:
        raise SystemExit("report.json missing 'k'")

    # Read metrics keys like "Recall@20"
    recall_key = f"Recall@{k}"
    mrr_key = f"MRR@{k}"
    hit_key = f"Hit@{k}"

    recall = float(rep.get(recall_key, 0.0))
    mrr = float(rep.get(mrr_key, 0.0))
    hit = float(rep.get(hit_key, 0.0))

    # thresholds (env override)
    min_recall = float(os.getenv("MIN_RECALL", "0.85"))
    min_mrr = float(os.getenv("MIN_MRR", "0.60"))
    min_hit = float(os.getenv("MIN_HIT", "0.85"))

    print(f"Eval metrics: {hit_key}={hit:.3f} {recall_key}={recall:.3f} {mrr_key}={mrr:.3f}")
    print(f"Thresholds:  min_hit={min_hit} min_recall={min_recall} min_mrr={min_mrr}")

    if hit < min_hit or recall < min_recall or mrr < min_mrr:
        raise SystemExit("❌ Retrieval quality gate FAILED")
    print("✅ Retrieval quality gate PASSED")

if __name__ == "__main__":
    main()
