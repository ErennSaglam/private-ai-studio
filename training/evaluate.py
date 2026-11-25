"""Generate answers for the test set and score them against the reference answers.

The reference project only reported test loss and perplexity. Loss says how likely the reference
tokens are; it does not say whether the generated answer is right. This script adds answer-level
metrics, computed on greedy (temperature 0) generations so that runs are comparable:

  exact match   normalised prediction == normalised reference
  token F1      bag-of-words overlap (SQuAD style)
  ROUGE-L F1    longest common word subsequence
  length ratio  predicted words / reference words

Usage (from the project root):
    python training/evaluate.py --label base
    python training/evaluate.py --label lora --adapter adapters/law-qa-r8-best
    python training/evaluate.py --label rag --rag-k 3          # BM25 retrieval, see training/rag.py

With --rag-k, retrieval quality is reported as answer support@k: the share of questions for which
at least one retrieved passage overlaps the reference answer with token F1 >= 0.5. (A "same source
block" hit cannot be used: the grouped split keeps whole test blocks out of the corpus.)
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]


def tr_lower(text: str) -> str:
    # Python's lower() maps "I" to "i"; Turkish needs "I" -> "ı" and "İ" -> "i".
    return text.replace("I", "ı").replace("İ", "i").lower()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", tr_lower(text))


def token_f1(pred: list[str], ref: list[str]) -> float:
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def rouge_l(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for a in pred:
        cur = [0]
        for j, b in enumerate(ref):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(ref)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of this run, e.g. base or lora")
    ap.add_argument("--model", default=ROOT / "models/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--adapter", default=None, help="adapter folder (adapters.safetensors + adapter_config.json)")
    ap.add_argument("--data", default=ROOT / "data/law-qa/test.jsonl")
    ap.add_argument("--out", default=ROOT / "training/results/law-qa")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rag-k", type=int, default=0, help="retrieve k passages with BM25 (0 = no RAG)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load(str(args.model), adapter_path=str(args.adapter) if args.adapter else None)
    sampler = make_sampler(temp=0.0)  # greedy: same input -> same output
    records = [json.loads(line) for line in open(args.data, encoding="utf-8")]

    retriever = None
    if args.rag_k:
        from rag import Retriever, rag_messages  # same folder

        retriever = Retriever()

    rows = []
    for i, rec in enumerate(records):
        msgs = rec["messages"]
        passages = []
        if retriever:
            passages = retriever.retrieve(msgs[-2]["content"], k=args.rag_k)
            msgs = rag_messages(msgs[:-1], passages) + [msgs[-1]]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)
        reference = msgs[-1]["content"]
        text, last = "", None
        start = time.perf_counter()
        for chunk in stream_generate(model, tok, prompt, max_tokens=args.max_tokens, sampler=sampler):
            text += chunk.text
            last = chunk
        elapsed = time.perf_counter() - start
        pred_w, ref_w = words(text), words(reference)
        rows.append({
            "id": i,
            "question": msgs[-2]["content"],
            "reference": reference,
            "prediction": text.strip(),
            "exact_match": float(pred_w == ref_w),
            "token_f1": round(token_f1(pred_w, ref_w), 4),
            "rouge_l": round(rouge_l(pred_w, ref_w), 4),
            "length_ratio": round(len(pred_w) / max(1, len(ref_w)), 3),
            "generated_tokens": last.generation_tokens,
            "hit_max_tokens": last.generation_tokens >= args.max_tokens,
            "tokens_per_second": round(last.generation_tps, 1),
            "seconds": round(elapsed, 2),
            "peak_memory_gb": round(last.peak_memory, 3),
            "prompt_tokens": last.prompt_tokens,
        })
        if retriever:
            support = max(token_f1(words(p["text"]), ref_w) for p in passages)
            rows[-1]["retrieved"] = [{"id": p["id"], "score": p["score"]} for p in passages]
            rows[-1]["retrieval_support_f1"] = round(support, 4)
            rows[-1]["retrieval_hit"] = float(support >= 0.5)
        print(f"[{i + 1}/{len(records)}] F1={rows[-1]['token_f1']:.2f} ROUGE-L={rows[-1]['rouge_l']:.2f}", flush=True)

    mean = lambda k: round(statistics.mean(r[k] for r in rows), 4)
    summary = {
        "label": args.label,
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "n": len(rows),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "exact_match": mean("exact_match"),
        "token_f1": mean("token_f1"),
        "rouge_l": mean("rouge_l"),
        "length_ratio_median": statistics.median(r["length_ratio"] for r in rows),
        "generated_tokens_mean": mean("generated_tokens"),
        "hit_max_tokens": sum(r["hit_max_tokens"] for r in rows),
        "tokens_per_second_median": statistics.median(r["tokens_per_second"] for r in rows),
        "seconds_total": round(sum(r["seconds"] for r in rows), 1),
        "peak_memory_gb_max": max(r["peak_memory_gb"] for r in rows),
        "prompt_tokens_mean": mean("prompt_tokens"),
        "rag_k": args.rag_k,
        "retrieval_hit_rate": mean("retrieval_hit") if retriever else None,
    }
    with open(out / f"predictions_{args.label}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"summary_{args.label}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
