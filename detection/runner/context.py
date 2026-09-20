from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from detection.data.chat import load_chat_pairs
from detection.data.challenges import ChallengeInstance, normalize_challenges
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
    challenges: list[ChallengeInstance] = field(
        default_factory=lambda: [ChallengeInstance(type="uniform", name="uniform")]
    )
    eval_corpus: str | None = None
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
    challenges = normalize_challenges(raw.get("challenges", ["uniform"]))
    return ExperimentSpec(**{**raw, "models": models, "pairs": pairs, "challenges": challenges})


class Context:
    """Shared state for a run: model paths, tokenizer, eval corpus, and model cache."""

    def __init__(self, root: Path, device: str = "auto", spec: ExperimentSpec | None = None) -> None:
        self.root = root
        self.device = device
        self.spec = spec
        self._cache: dict[str, tuple] = {}
        self._corpus_cache: dict[str, list[int]] = {}
        self._chat_cache: dict[str, list[dict[str, str]]] = {}

        tokenizer_name = spec.tokenizer_name if spec else None
        tokenizer_path = spec.tokenizer_path if spec else "tokenizer/tokenizer.json"
        tok_path = root / tokenizer_path
        # Load tokenizer matching model vocabulary.
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=HF_CACHE_DIR)
        elif tok_path.exists():
            self.tok: CharTokenizer | None = CharTokenizer.load(str(tok_path))
        else:
            self.tok = None

    def get_corpus_ids(self, path: str | None = None) -> list[int]:
        """Token ids for a text_window corpus, lazily loaded and cached.

        `path` defaults to the spec's `eval_corpus` (or corpora/eval_corpus.txt).
        """
        if self.tok is None:
            raise ValueError("text_window challenges require a tokenizer (tokenizer_name or tokenizer_path)")
        if path is None:
            path = (self.spec.eval_corpus if self.spec else None) or "corpora/eval_corpus.txt"
        if path not in self._corpus_cache:
            text = load_text(str(self.root / path))
            self._corpus_cache[path] = self.tok.encode(text)
        return self._corpus_cache[path]

    def get_chat_pairs(self, path: str) -> list[dict[str, str]]:
        """Chat challenge pool for a path, lazily loaded and cached."""
        if path not in self._chat_cache:
            self._chat_cache[path] = load_chat_pairs(self.root / path)
        return self._chat_cache[path]

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
