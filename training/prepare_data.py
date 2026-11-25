"""Turn the raw Turkish Law QA parquet into MLX-LM train/valid/test JSONL files.

Mirrors the reference project's csv_to_jsonl.py (Question/Answer -> chat messages, 80/10/10 split,
seed 42) and adds the checks it lacked: cleaning, exact de-duplication, a grouped split against
topic leakage, a near-duplicate leakage check and token-length statistics.

Usage (from the project root):
    python training/prepare_data.py
"""
import argparse
import json
import random
import re
import statistics
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "Sen Türk hukuku konusunda uzman bir asistansın. "
    "Soruları Türk mevzuatına ve yargı kararlarına dayanarak kısa, açık ve Türkçe cevapla."
)

# Rows that point the reader to another article instead of answering ("daha fazla bilgi ... yazıdan").
META_PATTERN = re.compile(r"daha fazla bilgi|isimli yazı|makalede|yazıda|bu yazı", re.IGNORECASE)


def normalize(text: str) -> str:
    """Unicode NFC + collapsed whitespace. Keeps the original casing for training."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def key(text: str) -> str:
    """Comparison key for duplicate detection (case-insensitive)."""
    return normalize(text).casefold()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    t = key(text)
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def to_record(row) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row.question},
            {"role": "assistant", "content": row.answer},
        ]
    }


def length_stats(values: list[int]) -> dict:
    values = sorted(values)
    pick = lambda q: values[int(q * (len(values) - 1))]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "p50": pick(0.5),
        "p95": pick(0.95),
        "max": values[-1],
        "sum": sum(values),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=ROOT / "data/raw/turkish_law_qa/train-00000-of-00001.parquet")
    ap.add_argument("--out", default=ROOT / "data/law-qa")
    ap.add_argument("--n-total", type=int, default=500, help="records to keep (reference used ~500)")
    ap.add_argument("--block", type=int, default=10, help="consecutive rows per topic group")
    ap.add_argument("--seed", type=int, default=42, help="same seed as the reference split")
    ap.add_argument("--leak-threshold", type=float, default=0.8, help="char-3gram Jaccard for near-duplicates")
    ap.add_argument("--tokenizer", default=ROOT / "models/Llama-3.2-1B-Instruct-4bit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"seed": args.seed, "system_prompt": SYSTEM_PROMPT}

    # 1) Load and clean ---------------------------------------------------------------------------
    df = pd.read_parquet(args.input)
    report["raw_rows"] = len(df)
    df["question"] = df["question"].map(normalize)
    df["answer"] = df["answer"].map(normalize)
    df["source_row"] = range(len(df))  # original order = source order (topics are contiguous)

    meta = df["question"].str.contains(META_PATTERN) | df["answer"].str.contains(META_PATTERN)
    df = df[~meta]
    report["dropped_meta_rows"] = int(meta.sum())

    # 2) Exact de-duplication (same question → keep the first answer) ------------------------------
    before = len(df)
    df = df[~df["question"].map(key).duplicated()]
    report["dropped_duplicate_questions"] = before - len(df)
    report["clean_rows"] = len(df)

    # 3) Grouped sampling and split ---------------------------------------------------------------
    # Neighbouring rows come from the same source article. Splitting rows at random would put
    # sibling questions about the same fact into train and test (topic leakage). We therefore
    # split whole blocks of consecutive source rows.
    df["group"] = df["source_row"] // args.block
    groups = [g for g, rows in df.groupby("group") if len(rows) == args.block]  # full blocks only
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_groups = args.n_total // args.block
    chosen = groups[:n_groups]
    n_train, n_valid = int(n_groups * 0.8), int(n_groups * 0.1)
    split_groups = {
        "train": chosen[:n_train],
        "valid": chosen[n_train : n_train + n_valid],
        "test": chosen[n_train + n_valid :],
    }
    splits = {name: df[df["group"].isin(gs)].copy() for name, gs in split_groups.items()}
    report["groups"] = {name: len(gs) for name, gs in split_groups.items()}

    # 4) Near-duplicate leakage check: valid/test questions vs. train questions --------------------
    train_grams = [char_ngrams(q) for q in splits["train"]["question"]]
    leaks = []
    for name in ("valid", "test"):
        for q in splits[name]["question"]:
            best = max(jaccard(char_ngrams(q), t) for t in train_grams)
            if best >= args.leak_threshold:
                leaks.append({"split": name, "question": q, "jaccard": round(best, 3)})
    report["near_duplicate_leaks"] = leaks
    # A leaked evaluation question would measure memorisation, not generalisation: drop it.
    for leak in leaks:
        part = splits[leak["split"]]
        splits[leak["split"]] = part[part["question"] != leak["question"]]

    # 5) Write CSV (reference format) and JSONL (MLX-LM chat format) -------------------------------
    for name, part in splits.items():
        part = part.sample(frac=1, random_state=args.seed)  # shuffle rows inside the split
        part[["source_row", "group", "question", "answer"]].rename(
            columns={"question": "Question", "answer": "Answer"}
        ).to_csv(out / f"{name}.csv", index=False)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in part.itertuples():
                f.write(json.dumps(to_record(row), ensure_ascii=False) + "\n")
        splits[name] = part
    report["rows"] = {name: len(part) for name, part in splits.items()}

    # 6) Token statistics with the real chat template ---------------------------------------------
    from mlx_lm import load  # imported late: loading the tokenizer is the slow part

    _, tok = load(str(args.tokenizer))
    report["tokens"] = {}
    for name, part in splits.items():
        full, prompt, answer = [], [], []
        for row in part.itertuples():
            msgs = to_record(row)["messages"]
            n_full = len(tok.apply_chat_template(msgs))
            # Same offset rule MLX-LM uses for --mask-prompt (tuner/datasets.py, ChatDataset).
            n_prompt = len(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True))
            full.append(n_full)
            prompt.append(n_prompt)
            answer.append(n_full - n_prompt)
        report["tokens"][name] = {
            "full": length_stats(full),
            "prompt": length_stats(prompt),
            "answer": length_stats(answer),
        }

    (out / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "tokens"}, ensure_ascii=False, indent=2))
    for name, t in report["tokens"].items():
        print(f"{name:5} full={t['full']}  answer={t['answer']}")


if __name__ == "__main__":
    main()
