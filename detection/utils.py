from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch

# All model artifacts (trained checkpoints and downloaded HF models) live under
# models_checkpoint/ at the monorepo root, shared with variant_training/.
MONOREPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CHECKPOINT_DIR = MONOREPO_ROOT / "models_checkpoint"
DETECTION_CHECKPOINT_DIR = MODEL_CHECKPOINT_DIR / "detection"
HF_CACHE_DIR = MODEL_CHECKPOINT_DIR / "hf_cache"


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }
