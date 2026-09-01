from __future__ import annotations

import torch


def top1_agreement(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """Fraction of token positions and sequences where argmax(ref) == argmax(cand)."""
    ref_top = ref_logits.argmax(dim=-1)
    cand_top = cand_logits.argmax(dim=-1)
    token_match = (ref_top == cand_top).float()
    seq_match = token_match.all(dim=-1).float() if ref_logits.dim() == 3 else token_match
    return {
        "token_agreement": float(token_match.mean()),
        "seq_agreement": float(seq_match.mean()),
    }


def exact_match(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> dict[str, float]:
    """Fraction of sequences where ref and cand logits are bitwise identical."""
    exact = (ref_logits == cand_logits).all(dim=-1).all(dim=-1).float()
    return {"seq_agreement": float(exact.mean())}
