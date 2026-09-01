from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import torch


def _checkpoint_root() -> Path:
    """Root that holds the shared models_checkpoint/ tree.

    Resolved against the *main* checkout (via ``git rev-parse --git-common-dir``)
    rather than the current working tree, so every git worktree points at one
    shared tree instead of downloading its own ~1.7 GB copy. Falls back to the
    repo root inferred from this file when git is unavailable (fresh tarball,
    some CI). Override the whole location with ``MODELS_CHECKPOINT_DIR``.
    """
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        # git prints this relative to `here` (e.g. "../.git") or absolute;
        # `here / common` handles both.
        return (here / common).resolve().parent
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return here.parent


# All model artifacts (trained checkpoints and downloaded HF models) live under
# a single models_checkpoint/ tree, shared with variants_training/ and across
# git worktrees.
MODEL_CHECKPOINT_DIR = Path(
    os.environ.get("MODELS_CHECKPOINT_DIR") or _checkpoint_root() / "models_checkpoint"
).resolve()
VARIANTS_CHECKPOINT_DIR = MODEL_CHECKPOINT_DIR / "variants"
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
