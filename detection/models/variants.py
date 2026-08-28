from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from detection.utils import ensure_dir, set_seed, write_json
from .loader import ModelConfig, TinyTransformerLM


def build_model(cfg: ModelConfig, seed: int, device: str) -> TinyTransformerLM:
    set_seed(seed)
    return TinyTransformerLM(cfg).to(device)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str | Path, model: nn.Module, cfg: ModelConfig, meta: dict[str, Any]
) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {"kind": "checkpoint", "state_dict": model.state_dict(), "model_config": asdict(cfg), "meta": meta},
        path,
    )


def save_quantized_spec(path: str | Path, source_checkpoint: str | Path) -> None:
    ensure_dir(Path(path).parent)
    write_json(path, {"kind": "dynamic_quantized_from_checkpoint", "source_checkpoint": Path(source_checkpoint).name})


def make_pruned_model(model: nn.Module, amount: float) -> nn.Module:
    """Prune linear and GPT-2 projection weights."""
    from transformers.pytorch_utils import Conv1D

    model = clone_model(model).cpu()
    targets = [
        (mod, "weight")
        for mod in model.modules()
        if isinstance(mod, (nn.Linear, Conv1D))
    ]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in targets:
        prune.remove(mod, name)
    return model.eval()


def train_model(
    cfg: ModelConfig,
    token_ids: list[int],
    seed: int,
    device: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    start_model: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    from detection.data.challenges import sample_lm_batch

    set_seed(seed)
    model = build_model(cfg, seed, device) if start_model is None else start_model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    for step in range(1, steps + 1):
        x, y = sample_lm_batch(token_ids, batch_size, seq_len, device, seed=seed * 10000 + step)
        loss = model.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    model.eval()
    return model, {
        "seed": seed, "steps": steps, "batch_size": batch_size,
        "seq_len": seq_len, "lr": lr,
        "final_loss": losses[-1], "mean_loss": sum(losses) / len(losses),
    }
