from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic

from detection.utils import HF_CACHE_DIR, MODEL_CHECKPOINT_DIR, get_device, read_json


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 192
    nhead: int = 4
    num_layers: int = 3
    dim_ff: int = 512
    max_len: int = 128
    dropout: float = 0.0


@dataclass
class LocalModelEntry:
    """Experiment-spec model entry pointing at a local checkpoint (.pt) or spec (.json),
    resolved against VARIANTS_CHECKPOINT_DIR if relative."""

    path: str
    kind: str = "local"


@dataclass
class HFModelEntry:
    """Experiment-spec model entry describing a HuggingFace causal LM."""

    model_id: str
    kind: str = "hf_model"
    revision: str | None = None
    adapter_path: str | None = None
    dtype: str = "float32"


ModelEntry = LocalModelEntry | HFModelEntry


class GPT2Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits


class HFWrapper(GPT2Wrapper):
    """Any HF causal LM (optionally with a PEFT adapter), exposed as ids -> logits."""


def load_hf_spec(
    spec: HFModelEntry, preferred_device: str = "auto"
) -> tuple[nn.Module, ModelConfig, dict[str, Any], str]:
    """Load an HF causal LM described by an HFModelEntry (optionally with a PEFT adapter;
    adapter_path resolves inside models_checkpoint/ if relative)."""
    from transformers import AutoModelForCausalLM

    device = get_device(preferred_device)
    dtype = getattr(torch, spec.dtype)
    inner = AutoModelForCausalLM.from_pretrained(
        spec.model_id, revision=spec.revision, dtype=dtype, cache_dir=HF_CACHE_DIR
    )
    meta: dict[str, Any] = {"variant": "hf_base", "model_id": spec.model_id}
    if spec.adapter_path:
        from peft import PeftModel

        adapter_dir = Path(spec.adapter_path)
        if not adapter_dir.is_absolute():
            adapter_dir = MODEL_CHECKPOINT_DIR / adapter_dir
        inner = PeftModel.from_pretrained(inner, str(adapter_dir))
        meta = {**meta, "variant": "hf_peft_adapter", "adapter_path": str(adapter_dir)}
    hf_cfg = inner.config
    cfg = ModelConfig(
        vocab_size=hf_cfg.vocab_size,
        d_model=hf_cfg.hidden_size,
        nhead=hf_cfg.num_attention_heads,
        num_layers=hf_cfg.num_hidden_layers,
        dim_ff=hf_cfg.intermediate_size,
        max_len=hf_cfg.max_position_embeddings,
        dropout=0.0,
    )
    model = HFWrapper(inner.eval())
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    return model, cfg, meta, device


def convert_gpt2_conv1d(model: nn.Module) -> nn.Module:
    """Convert GPT-2 projections before dynamic quantization."""
    from transformers.pytorch_utils import Conv1D

    for name, child in list(model.named_children()):
        if isinstance(child, Conv1D):
            linear = nn.Linear(child.weight.shape[0], child.weight.shape[1])
            with torch.no_grad():
                linear.weight.copy_(child.weight.T)
                linear.bias.copy_(child.bias)
            setattr(model, name, linear)
        else:
            convert_gpt2_conv1d(child)
    return model


def load_float_checkpoint(
    path: str, device: str = "cpu"
) -> tuple[nn.Module, ModelConfig, dict[str, Any]]:
    from transformers import GPT2LMHeadModel

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "checkpoint":
        raise ValueError(f"Expected checkpoint file, got {path}")
    cfg = ModelConfig(**payload["model_config"])
    if "gpt2" in str(path):
        inner = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=HF_CACHE_DIR)
        inner.load_state_dict(payload["state_dict"])
        model: nn.Module = GPT2Wrapper(inner)
    else:
        model = TinyTransformerLM(cfg)
        model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, cfg, payload.get("meta", {})


def load_model_any(
    path: str, preferred_device: str = "auto"
) -> tuple[nn.Module, ModelConfig, dict[str, Any], str]:
    """Unified loader: handles float .pt checkpoints and quantized .json specs."""
    device = get_device(preferred_device)
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        spec = read_json(path_obj)
        if spec.get("kind") == "hf_model":
            return load_hf_spec(HFModelEntry(**spec), preferred_device=preferred_device)
        if spec.get("kind") != "dynamic_quantized_from_checkpoint":
            raise ValueError(f"Unknown model spec kind: {spec.get('kind')}")
        source = str((path_obj.parent / spec["source_checkpoint"]).resolve())
        base_model, cfg, meta = load_float_checkpoint(source, device="cpu")
        if isinstance(base_model, GPT2Wrapper):
            base_model = convert_gpt2_conv1d(base_model)
        q_model = quantize_dynamic(base_model, {nn.Linear}, dtype=torch.qint8)
        q_meta = {**meta, "variant": "dynamic_int8", "source_checkpoint": source}
        return q_model, cfg, q_meta, "cpu"
    model, cfg, meta = load_float_checkpoint(path, device=device)
    return model, cfg, meta, device


# ---------------------------------------------------------------------------
# TinyTransformerLM: a small from-scratch GPT-style model, only instantiated
# by load_float_checkpoint for non-GPT-2 checkpoints. Not used by any HF path
# (load_hf_spec, or "gpt2" checkpoints, which wrap a real HF model instead).
# ---------------------------------------------------------------------------


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.nhead, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyCausalSelfAttention(d_model, nhead)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(cfg.d_model, cfg.nhead, cfg.dim_ff) for _ in range(cfg.num_layers)]
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"seq_len {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            self(input_ids).reshape(-1, self.cfg.vocab_size), targets.reshape(-1)
        )
