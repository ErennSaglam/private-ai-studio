"""Build training/results/law-qa/report.md from experiments.json, the curves and the predictions.

Usage (from the project root):
    python training/report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "training/results/law-qa"


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def curve_table(curve: dict) -> list[str]:
    val = {int(k): v for k, v in curve["val"].items()}
    train = {int(k): v["loss"] for k, v in curve["train"].items()}
    lines = ["| iter | train loss (son 10 adım ort.) | val loss |", "|---|---|---|"]
    for it in sorted(val):
        lines.append(f"| {it} | {fmt(train.get(it))} | {fmt(val[it])} |")
    return lines


def main() -> None:
    exp = json.loads((RESULTS / "experiments.json").read_text())
    out = ["# Law-QA LoRA deney sonuçları", "", "Otomatik üretildi: `python training/report.py`. Tüm değerler bu makinede ölçüldü.", ""]

    c = exp.get("control")
    if c:
        out += ["## 1. Kontrol eğitimi (rank 8, lr 1e-5, 16 katman, 600 iter)", ""]
        curve = json.loads((RESULTS / "curve_control.json").read_text())
        out += curve_table(curve) + [""]
        out += [
            f"- En iyi checkpoint: **iter {c['curve']['best']['iter']}** (val loss {c['curve']['best']['val_loss']:.3f})",
            f"- Son iterasyon: iter {c['curve']['last']['iter']} (val loss {c['curve']['last']['val_loss']:.3f})",
            "",
            "| Model | Test loss | Test PPL |",
            "|---|---|---|",
            f"| Base (adapter yok) | {fmt(c['test_base']['test_loss'])} | {fmt(c['test_base']['test_ppl'], 2)} |",
            f"| LoRA — son iter | {fmt(c['test_last']['test_loss'])} | {fmt(c['test_last']['test_ppl'], 2)} |",
            f"| LoRA — en iyi checkpoint | {fmt(c['test_best']['test_loss'])} | {fmt(c['test_best']['test_ppl'], 2)} |",
            "",
            "## 2. Sistem karşılaştırması (aynı 50 test sorusu, temperature 0)",
            "",
            "| Sistem | Token F1 | ROUGE-L | Exact match | Uzunluk oranı (medyan) | Retrieval hit | Prompt token (ort.) | tok/s (medyan) | Peak mem (GB) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        names = {"base": "Base", "lora": "LoRA", "rag": "RAG (BM25, k=3)", "lora_rag": "LoRA + RAG"}
        for key, label in names.items():
            s = c["eval"][key]
            out.append(
                f"| {label} | {fmt(s['token_f1'])} | {fmt(s['rouge_l'])} | {fmt(s['exact_match'], 2)} | "
                f"{fmt(s['length_ratio_median'], 2)} | {fmt(s['retrieval_hit_rate'], 2)} | {fmt(s['prompt_tokens_mean'], 0)} | "
                f"{fmt(s['tokens_per_second_median'], 1)} | {fmt(s['peak_memory_gb_max'], 2)} |"
            )
        out.append("")

        # Side-by-side examples
        preds = {k: [json.loads(l) for l in open(RESULTS / f"predictions_{k}.jsonl", encoding="utf-8")] for k in names}
        out += ["## 3. Örnek cevaplar (ilk 5 test sorusu)", ""]
        for i in range(min(5, len(preds["base"]))):
            out += [f"**Soru {i + 1}:** {preds['base'][i]['question']}", "", f"- **Referans:** {preds['base'][i]['reference']}"]
            for key, label in names.items():
                out.append(f"- **{label}:** {preds[key][i]['prediction']}")
            out.append("")

    series = {k: v for k, v in exp.items() if k != "control"}
    if series:
        out += [
            "## 4. Hiperparametre serisi (tek değişken; diğer her şey kontrol konfigürasyonu)",
            "",
            "| Deney | Değişen | En iyi iter | Val loss (min) | Test loss | Test PPL | Token F1 | ROUGE-L | Eğitim süresi |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        if c:
            out.append(
                f"| control | — | {c['curve']['best']['iter']} | {fmt(c['curve']['best']['val_loss'])} | {fmt(c['test_best']['test_loss'])} | "
                f"{fmt(c['test_best']['test_ppl'], 2)} | {fmt(c['eval']['lora']['token_f1'])} | {fmt(c['eval']['lora']['rouge_l'])} | — |"
            )
        for name, r in series.items():
            out.append(
                f"| {name} | `{json.dumps(r['change'])}` | {r['curve']['best']['iter']} | {fmt(r['curve']['best']['val_loss'])} | "
                f"{fmt(r['test_best']['test_loss'])} | {fmt(r['test_best']['test_ppl'], 2)} | {fmt(r['eval']['token_f1'])} | "
                f"{fmt(r['eval']['rouge_l'])} | {r['train_seconds'] / 60:.1f} dk |"
            )
        out.append("")

    (RESULTS / "report.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
