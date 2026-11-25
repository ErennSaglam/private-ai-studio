import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


class ModelService:
    def __init__(self, model_path: Path, adapter_path: Path | None = None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._stream = None
        # MLX GPU streams belong to the thread that created them, and one model must not
        # generate in parallel. A single dedicated worker thread solves both: every MLX
        # call runs there, and requests queue up one after another.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def load(self) -> None:
        self._executor.submit(self._load).result()

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        return self._executor.submit(self._chat, messages, max_tokens, temperature).result()

    def _load(self) -> None:
        self._stream = mx.new_stream(mx.default_device())
        self.model, self.tokenizer = load(
            str(self.model_path),
            adapter_path=str(self.adapter_path) if self.adapter_path else None,
        )

    def _chat(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampler = make_sampler(temp=temperature)

        text, last = "", None
        start = time.perf_counter()
        for chunk in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            stream=self._stream,
        ):
            text += chunk.text
            last = chunk
        elapsed = time.perf_counter() - start

        return {
            "reply": text.strip(),
            "prompt_tokens": last.prompt_tokens,
            "generated_tokens": last.generation_tokens,
            "tokens_per_second": round(last.generation_tps, 2),
            "elapsed_seconds": round(elapsed, 2),
            "peak_memory_gb": round(last.peak_memory, 3),
        }

    def info(self) -> dict:
        config = json.loads((self.model_path / "config.json").read_text())
        size_bytes = sum(f.stat().st_size for f in self.model_path.glob("*.safetensors"))
        return {
            "model_path": str(self.model_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "num_layers": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "num_attention_heads": config["num_attention_heads"],
            "vocab_size": config["vocab_size"],
            "quantization": config.get("quantization"),
            "weights_size_gb": round(size_bytes / 1e9, 3),
        }
