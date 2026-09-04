from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from detection.data.tokenizer import CharTokenizer, load_text
from detection.models.loader import HFModelEntry, LocalModelEntry, ModelEntry, load_hf_spec, load_model_any
from detection.utils import VARIANTS_CHECKPOINT_DIR, HF_CACHE_DIR, read_json


@dataclass
class PairSpec:
    ref: str
    cand: str
    name: str | None = None
    wrap_cand: str | None = None


@dataclass
class ExperimentSpec:
    name: str
    models: dict[str, ModelEntry]
    pairs: list[PairSpec]
    description: str | None = None
    outdir: str | None = None
    tokenizer_name: str | None = None
    tokenizer_path: str = "tokenizer/tokenizer.json"
    metrics: list[str] | None = None
    distributions: list[str] = field(default_factory=lambda: ["uniform"])
    k_values: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    repeats: int = 5
    seq_len: int = 64
    batch_size: int = 32
    decision_metric: str | None = None
    beta: float = 0.05
    reject_threshold: float | None = None
    reject_on: str = "token"
    reject_rate_threshold: float = 0.95


_MODEL_ENTRY_KINDS = {"local": LocalModelEntry, "hf_model": HFModelEntry}


def _parse_model_entry(raw: dict[str, Any]) -> ModelEntry:
    kind = raw.get("kind")
    if kind not in _MODEL_ENTRY_KINDS:
        raise ValueError(f"Unknown model entry kind: {kind!r}. Available: {list(_MODEL_ENTRY_KINDS)}")
    return _MODEL_ENTRY_KINDS[kind](**raw)


def parse_experiment_spec(path: str | os.PathLike[str]) -> ExperimentSpec:
    raw = read_json(path)
    models = {key: _parse_model_entry(entry) for key, entry in raw["models"].items()}
    pairs = [PairSpec(**p) for p in raw["pairs"]]
    return ExperimentSpec(**{**raw, "models": models, "pairs": pairs})


class Context:
    """Shared state for a run: model paths, tokenizer, eval corpus, and model cache."""

    def __init__(self, root: Path, device: str = "auto", spec: ExperimentSpec | None = None) -> None:
        self.root = root
        self.device = device
        self.spec = spec
        self._cache: dict[str, tuple] = {}

        tokenizer_name = spec.tokenizer_name if spec else None
        tokenizer_path = spec.tokenizer_path if spec else "tokenizer/tokenizer.json"
        tok_path = root / tokenizer_path
        # Load tokenizer matching model vocabulary.
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=HF_CACHE_DIR)
            self.eval_ids = self.tok.encode(
                load_text(str(root / "corpora" / "eval_corpus.txt"))
            )
        elif tok_path.exists():
            self.tok: CharTokenizer | None = CharTokenizer.load(str(tok_path))
            self.eval_ids: list[int] | None = self.tok.encode(
                load_text(str(root / "corpora" / "eval_corpus.txt"))
            )
        else:
            self.tok = None
            self.eval_ids = None

    def _load(self, key: str) -> tuple:
        models = self.spec.models if self.spec else {}
        if key not in models:
            raise KeyError(f"Model '{key}' not in experiment spec. Available: {list(models)}")
        entry = models[key]
        # HF entries describe HF models (base or base + PEFT adapter).
        if isinstance(entry, HFModelEntry):
            cache_key = json.dumps(asdict(entry), sort_keys=True)
            if cache_key not in self._cache:
                self._cache[cache_key] = load_hf_spec(entry, preferred_device=self.device)
            return self._cache[cache_key]
        # Local entries: relative paths resolve inside the shared models_checkpoint/ tree.
        assert isinstance(entry, LocalModelEntry)
        path = Path(entry.path)
        if not path.is_absolute():
            path = VARIANTS_CHECKPOINT_DIR / path
        path = str(path.resolve())
        if path not in self._cache:
            self._cache[path] = load_model_any(path, preferred_device=self.device)
        return self._cache[path]

    def get_pair(self, ref_key: str, cand_key: str) -> dict:
        ref_model, ref_cfg, ref_meta, ref_device = self._load(ref_key)
        cand_model, cand_cfg, cand_meta, cand_device = self._load(cand_key)
        if ref_cfg != cand_cfg:
            raise ValueError(f"Config mismatch: {ref_key} vs {cand_key}")
        eval_device = "cpu" if (ref_device == "cpu" or cand_device == "cpu") else ref_device
        ref_model.to(eval_device).eval()
        try:
            cand_model.to(eval_device).eval()
        except Exception:
            eval_device = "cpu"
            ref_model.to("cpu").eval()
            cand_model.to("cpu").eval()
        return {
            "ref_key": ref_key,
            "cand_key": cand_key,
            "ref_model": ref_model,
            "cand_model": cand_model,
            "cfg": ref_cfg,
            "ref_meta": ref_meta,
            "cand_meta": cand_meta,
            "eval_device": eval_device,
        }
