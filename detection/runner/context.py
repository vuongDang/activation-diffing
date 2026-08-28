from __future__ import annotations

from pathlib import Path

import json

from detection.data.tokenizer import CharTokenizer, load_text
from detection.models.loader import load_hf_spec, load_model_any
from detection.utils import DETECTION_CHECKPOINT_DIR, HF_CACHE_DIR


class Context:
    """Shared state for a run: model paths, tokenizer, eval corpus, and model cache."""

    def __init__(self, root: Path, device: str = "auto", spec: dict | None = None) -> None:
        self.root = root
        self.device = device
        self._spec = spec or {}
        self._cache: dict[str, tuple] = {}

        tokenizer_name = self._spec.get("tokenizer_name")
        tok_path = root / self._spec.get("tokenizer_path", "tokenizer/tokenizer.json")
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

    def model_paths(self) -> dict[str, str]:
        return self._spec.get("models", {})

    def _load(self, key: str) -> tuple:
        paths = self.model_paths()
        if key not in paths:
            raise KeyError(f"Model '{key}' not in assets_manifest. Available: {list(paths)}")
        entry = paths[key]
        # Inline dict entries describe HF models (base or base + PEFT adapter).
        if isinstance(entry, dict):
            cache_key = json.dumps(entry, sort_keys=True)
            if cache_key not in self._cache:
                self._cache[cache_key] = load_hf_spec(entry, preferred_device=self.device)
            return self._cache[cache_key]
        # Relative model paths resolve inside the shared models_checkpoint/ tree.
        path = Path(entry)
        if not path.is_absolute():
            path = DETECTION_CHECKPOINT_DIR / path
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
