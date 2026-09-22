from __future__ import annotations

import torch
import torch.nn.functional as F


def max_abs_diff(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """Largest single-dimension activation change. Catches a localized blow-up
    (one feature/neuron moving a lot) that a mean-based metric would dilute."""
    diff = (ref_h - cand_h).abs()
    per_position_max = diff.amax(dim=-1)
    return {
        "max": float(diff.amax()),
        "mean_of_position_max": float(per_position_max.mean()),
    }


def l2_distance(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """L2 distance per position, plus the same distance relative to the reference
    norm. Raw L2 isn't comparable across layers since residual-stream norms grow
    with depth; `relative_mean` is."""
    diff_norm = (ref_h - cand_h).norm(dim=-1)
    ref_norm = ref_h.norm(dim=-1)
    relative = diff_norm / ref_norm.clamp_min(1e-8)
    return {
        "mean": float(diff_norm.mean()),
        "relative_mean": float(relative.mean()),
    }


def norm_ratio(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """‖cand‖ / ‖ref‖ per position — pure magnitude scaling, independent of direction."""
    ref_norm = ref_h.norm(dim=-1)
    cand_norm = cand_h.norm(dim=-1)
    ratio = cand_norm / ref_norm.clamp_min(1e-8)
    return {"mean": float(ratio.mean())}


def cosine_similarity(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """Cosine similarity per position — did the activation rotate, independent of scale."""
    sim = F.cosine_similarity(ref_h, cand_h, dim=-1)
    return {"mean": float(sim.mean())}


def diff_direction_consistency(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """Cosine similarity between each position's diff vector (cand - ref) and the
    batch-mean diff direction. High values mean the drift looks like a single
    injected/steering direction; low values mean it's input-dependent."""
    diff = (cand_h - ref_h).reshape(-1, ref_h.shape[-1])
    mean_dir = diff.mean(dim=0, keepdim=True)
    sim = F.cosine_similarity(diff, mean_dir.expand_as(diff), dim=-1)
    return {"mean": float(sim.mean()), "mean_diff_norm": float(mean_dir.norm())}


def diff_effective_rank(ref_h: torch.Tensor, cand_h: torch.Tensor) -> dict[str, float]:
    """Stable rank (trace / top eigenvalue of the energy spectrum) of the centered
    per-position diff vectors, via SVD. Low rank looks like a single injected
    direction (LoRA/steering vector); high rank looks like broad, noisy drift
    (e.g. quantization). Computed per batch, so results are a batch-size-weighted
    approximation of the full-k statistic, not identical to computing it over all
    k challenges at once."""
    diff = (cand_h - ref_h).reshape(-1, ref_h.shape[-1]).float()
    if diff.shape[0] < 2:
        return {"stable_rank": 0.0, "top_singular_value": 0.0}
    s = torch.linalg.svdvals(diff - diff.mean(dim=0, keepdim=True))
    energy = s ** 2
    total = float(energy.sum())
    top = float(energy[0]) if energy.numel() else 0.0
    return {
        "stable_rank": total / top if top > 0 else 0.0,
        "top_singular_value": float(s[0]) if s.numel() else 0.0,
    }
