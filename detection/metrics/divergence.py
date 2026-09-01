from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """KL(ref || cand) averaged over token positions and sequences."""
    ref_log = F.log_softmax(ref_logits, dim=-1)
    cand_log = F.log_softmax(cand_logits, dim=-1)
    kl_tok = (ref_log.exp() * (ref_log - cand_log)).sum(dim=-1)
    return {"mean": float(kl_tok.mean())}


def tv_distance(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """Total variation distance averaged over token positions and sequences."""
    ref_probs = F.softmax(ref_logits, dim=-1)
    cand_probs = F.softmax(cand_logits, dim=-1)
    tv = 0.5 * (ref_probs - cand_probs).abs().sum(dim=-1)
    return {"mean": float(tv.mean())}


def l2_distance(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """L2 distance on raw logits averaged over token positions and sequences."""
    l2 = ((ref_logits - cand_logits) ** 2).sum(dim=-1)
    return {"mean": float(l2.mean())}
