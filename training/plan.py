"""Print the training calculations for an mlx_lm.lora YAML config, before training starts.

Every number follows the batching rules of mlx-lm 0.32.2 (tuner/trainer.py):
batches are built from length-sorted examples, N mod B examples are never used,
and validation uses min(floor(N_val / B), val_batches) batches.

Usage (from the project root):
    python training/plan.py training/configs/law-qa-lora.yaml
"""
import json
import math
import sys
from pathlib import Path

import yaml

DEFAULT_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_params_per_block(cfg: dict, rank: int, keys: list[str]) -> int:
    """r * (d_in + d_out) summed over the adapted linear layers of one transformer block."""
    d = cfg["hidden_size"]
    head_dim = cfg.get("head_dim") or d // cfg["num_attention_heads"]
    q_out = cfg["num_attention_heads"] * head_dim
    kv_out = cfg["num_key_value_heads"] * head_dim
    ff = cfg["intermediate_size"]
    shapes = {  # (d_in, d_out)
        "q_proj": (d, q_out), "k_proj": (d, kv_out), "v_proj": (d, kv_out), "o_proj": (q_out, d),
        "gate_proj": (d, ff), "up_proj": (d, ff), "down_proj": (ff, d),
    }
    return sum(rank * sum(shapes[k.split(".")[-1]]) for k in keys)


def total_params(cfg: dict) -> int:
    """Unquantized parameter count of a Llama-style model (embeddings tied or not)."""
    d, ff, v, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    head_dim = cfg.get("head_dim") or d // cfg["num_attention_heads"]
    q, kv = cfg["num_attention_heads"] * head_dim, cfg["num_key_value_heads"] * head_dim
    block = d * q + 2 * d * kv + q * d + 3 * d * ff + 2 * d
    head = 0 if cfg.get("tie_word_embeddings") else v * d
    return v * d + layers * block + d + head


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "training/configs/law-qa-lora.yaml")
    c = yaml.safe_load(config_path.read_text())
    stats = json.loads((Path(c["data"]) / "stats.json").read_text())
    mcfg = json.loads((Path(c["model"]) / "config.json").read_text())

    B, iters = c["batch_size"], c["iters"]
    k = c.get("grad_accumulation_steps", 1)
    n_train, n_val, n_test = stats["rows"]["train"], stats["rows"]["valid"], stats["rows"]["test"]
    tok = stats["tokens"]["train"]

    steps_per_epoch = n_train // B
    val_b = n_val // B if c["val_batches"] == -1 else min(n_val // B, c["val_batches"])
    test_b = n_test // B if c.get("test_batches", 500) == -1 else min(n_test // B, c.get("test_batches", 500))
    n_evals = 1 + iters // c["steps_per_eval"] + (1 if iters % c["steps_per_eval"] else 0)

    layers_total = mcfg["num_hidden_layers"]
    n_layers = layers_total if c["num_layers"] == -1 else c["num_layers"]
    lp = c["lora_parameters"]
    keys = lp.get("keys") or DEFAULT_KEYS
    p_lora = n_layers * lora_params_per_block(mcfg, lp["rank"], keys)
    p_total = total_params(mcfg)

    # Padding: each batch is padded to 1 + 32 * ceil(longest / 32) (capped by max_seq_length).
    t_pad_max = min(1 + 32 * math.ceil(tok["full"]["max"] / 32), c["max_seq_length"])
    logits_gb = B * t_pad_max * mcfg["vocab_size"] * 4 / 1e9

    rows = [
        ("Train / valid / test örnek", f"{n_train} / {n_val} / {n_test}"),
        ("Adım/epoch = floor(N_train / B)", f"floor({n_train}/{B}) = {steps_per_epoch}"),
        ("Kullanılmayan train örneği = N mod B", f"{n_train % B}"),
        ("Epoch = iters / adım_epoch", f"{iters}/{steps_per_epoch} = {iters / steps_per_epoch:.2f}"),
        ("Etkin batch = B × accum", f"{B} × {k} = {B * k}"),
        ("Optimizer güncellemesi = iters / accum", f"{iters // k}"),
        ("İşlenen örnek = iters × B", f"{iters * B}"),
        ("İşlenen token ≈ epoch × Σ token", f"{iters / steps_per_epoch:.2f} × {tok['full']['sum']} ≈ {iters / steps_per_epoch * tok['full']['sum']:,.0f}"),
        ("Loss'a giren token ≈ epoch × Σ cevap token", f"≈ {iters / steps_per_epoch * tok['answer']['sum']:,.0f} (mask_prompt={c.get('mask_prompt', False)})"),
        ("Validation batch", f"{val_b} batch = {val_b * B} örnek (her ölçümde)"),
        ("Validation ölçüm sayısı", f"{n_evals}"),
        ("Test batch", f"{test_b} batch = {test_b * B} örnek"),
        ("Checkpoint = floor(iters / save_every)", f"{iters // c['save_every']} + final adapters.safetensors"),
        ("Toplam katman / LoRA katmanı / rank", f"{layers_total} / {n_layers} / {lp['rank']}"),
        ("Hedef modüller", ", ".join(keys)),
        ("Eğitilebilir parametre", f"{p_lora:,} ({p_lora / p_total * 100:.3f}% / {p_total:,})"),
        ("Adapter dosyası (float32)", f"≈ {p_lora * 4 / 1e6:.1f} MB"),
        ("En uzun örnek / pad edilmiş uzunluk", f"{tok['full']['max']} / {t_pad_max} token"),
        ("Logits tensörü (B × T_pad × V × 4 bayt)", f"≈ {logits_gb:.2f} GB (teorik, en uzun batch)"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"Plan: {config_path}\n")
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


if __name__ == "__main__":
    main()
