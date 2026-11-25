"""List the tensors stored in a local MLX model and summarize its config."""
import json
import sys
from pathlib import Path

import mlx.core as mx

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models/Llama-3.2-1B-Instruct-4bit")
config = json.loads((model_dir / "config.json").read_text())

print("Layers      :", config["num_hidden_layers"])
print("Hidden size :", config["hidden_size"])
print("Heads       :", config["num_attention_heads"], "| KV heads:", config.get("num_key_value_heads"))
print("Vocab size  :", config["vocab_size"])
print("Quantization:", config.get("quantization"))

for file in sorted(model_dir.glob("*.safetensors")):
    tensors = mx.load(str(file))
    print(f"\n{file.name}: {len(tensors)} tensors")
    for name in sorted(tensors)[:12]:  # first layer is enough to see the pattern
        t = tensors[name]
        print(f"  {name:55} {str(t.shape):20} {t.dtype}")
