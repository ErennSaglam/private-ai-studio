import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-4bit"
# Optional LoRA adapter, chosen at start-up: ADAPTER=law-qa-r8-best uvicorn app.main:app ...
# Unset means the plain base model.
ADAPTER_PATH = PROJECT_ROOT / "adapters" / os.environ["ADAPTER"] if os.environ.get("ADAPTER") else None
FRONTEND_ORIGIN = "http://localhost:3000"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
