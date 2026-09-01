from __future__ import annotations

import torch
import torch.nn.functional as F


def token_difr(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """Token-DiFR gap, mismatch rate, and TV distance."""
    ref_log = F.log_softmax(ref_logits, dim=-1)
    ref_top = ref_logits.argmax(dim=-1)
    cand_top = cand_logits.argmax(dim=-1)
    ref_top_logp = ref_log.gather(-1, ref_top.unsqueeze(-1)).squeeze(-1)
    cand_top_logp_under_ref = ref_log.gather(-1, cand_top.unsqueeze(-1)).squeeze(-1)
    difr_gap = (-cand_top_logp_under_ref) - (-ref_top_logp)
    tv = 0.5 * (F.softmax(ref_logits, dim=-1) - F.softmax(cand_logits, dim=-1)).abs().sum(dim=-1)
    return {
        "difr_gap": float(difr_gap.mean()),
        "mismatch_rate": float((ref_top != cand_top).float().mean()),
        "tv_mean": float(tv.mean()),
    }
