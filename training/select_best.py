"""Read an mlx_lm.lora training log, pick the checkpoint with the lowest validation loss
and copy it into its own adapter folder (manual early stopping).

The reference project always used the adapter from the last iteration. If validation loss
rises at the end (overfitting), an earlier checkpoint generalises better.

Usage (from the project root):
    python training/select_best.py training/results/law-qa/train.log adapters/law-qa-r8
    -> adapters/law-qa-r8-best/ + training/results/law-qa/curve.json
"""
import json
import re
import shutil
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")
# mlx-lm 0.32.2 table row: "   210    1.159 ▼       49     37.7k"  (iter, train loss, trend, tok/s, trained tokens)
TRAIN = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s*[▲▼=]?\s+([\d.]+)\s+([\d.]+k?)\s*$")


def to_int(value: str) -> int:
    return int(float(value[:-1]) * 1000) if value.endswith("k") else int(float(value))
PEAK = re.compile(r"peak[^\d]*([\d.]+)\s*GB", re.IGNORECASE)


def parse(log_text: str) -> dict:
    lines = ANSI.sub("", log_text).replace("\r", "\n").splitlines()
    val, train, peaks = {}, {}, []
    for line in lines:
        if m := VAL.match(line):
            val[int(m[1])] = float(m[2])
        elif m := TRAIN.match(line):
            train[int(m[1])] = {"loss": float(m[2]), "tok_s": float(m[3]), "tokens": to_int(m[4])}
        if m := PEAK.search(line):
            peaks.append(float(m[1]))
    return {"val": val, "train": train, "peak_memory_gb": max(peaks) if peaks else None}


def main() -> None:
    log_path, adapter_dir = Path(sys.argv[1]), Path(sys.argv[2])
    curve = parse(log_path.read_text(errors="replace"))
    if not curve["val"]:
        sys.exit("No validation lines found in the log.")

    # Only iterations that have a saved checkpoint can be selected.
    saved = {int(p.name.split("_")[0]): p for p in adapter_dir.glob("*_adapters.safetensors")}
    candidates = {it: loss for it, loss in curve["val"].items() if it in saved}
    best_it = min(candidates, key=candidates.get)

    best_dir = adapter_dir.with_name(adapter_dir.name + "-best")
    best_dir.mkdir(exist_ok=True)
    shutil.copy(saved[best_it], best_dir / "adapters.safetensors")
    shutil.copy(adapter_dir / "adapter_config.json", best_dir / "adapter_config.json")

    curve["best"] = {"iter": best_it, "val_loss": candidates[best_it], "folder": str(best_dir)}
    curve["last"] = {"iter": max(curve["val"]), "val_loss": curve["val"][max(curve["val"])]}
    out = log_path.with_name("curve.json")
    out.write_text(json.dumps(curve, indent=2))
    print("val loss:", {k: curve["val"][k] for k in sorted(curve["val"])})
    print(f"best checkpoint: iter {best_it} (val loss {candidates[best_it]:.3f}) -> {best_dir}")
    print(f"last iteration: iter {curve['last']['iter']} (val loss {curve['last']['val_loss']:.3f})")


if __name__ == "__main__":
    main()
