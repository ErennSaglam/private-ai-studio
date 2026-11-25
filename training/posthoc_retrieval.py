"""Recompute answer support@k for saved RAG predictions without re-running the model.

Needed once: the first control evaluation used a "same source block" hit, which is always 0 with
the grouped split. The retrieved passage ids are stored in the predictions, so the new metric
(see evaluate.py) can be computed from them.

Usage (from the project root):
    python training/posthoc_retrieval.py rag lora_rag
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import token_f1, words  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "training/results/law-qa"
passages = {p["id"]: p["text"] for p in map(json.loads, open(ROOT / "data/law-qa/rag_corpus.jsonl", encoding="utf-8"))}

for label in sys.argv[1:]:
    pred_file = RESULTS / f"predictions_{label}.jsonl"
    rows = [json.loads(line) for line in open(pred_file, encoding="utf-8")]
    for r in rows:
        ref = words(r["reference"])
        r["retrieval_support_f1"] = round(max(token_f1(words(passages[p["id"]]), ref) for p in r["retrieved"]), 4)
        r["retrieval_hit"] = float(r["retrieval_support_f1"] >= 0.5)
    with open(pred_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary_file = RESULTS / f"summary_{label}.json"
    s = json.loads(summary_file.read_text())
    s["retrieval_hit_rate"] = round(statistics.mean(r["retrieval_hit"] for r in rows), 4)
    s["retrieval_support_f1_mean"] = round(statistics.mean(r["retrieval_support_f1"] for r in rows), 4)
    summary_file.write_text(json.dumps(s, ensure_ascii=False, indent=2))
    print(label, "answer support@k:", s["retrieval_hit_rate"], "mean support F1:", s["retrieval_support_f1_mean"])
