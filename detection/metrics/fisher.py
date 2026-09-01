from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fisher_diagonal(
    model: nn.Module,
    challenges: torch.Tensor,
    batch_size: int = 4,
) -> torch.Tensor:
    """Diagonal of the Fisher information matrix estimated via score samples."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]
    accum = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in params]
    n = challenges.shape[0]

    for i in range(0, n, batch_size):
        batch = challenges[i : i + batch_size]
        logits = model(batch)[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        sampled = torch.multinomial(log_probs.exp().detach(), num_samples=1).squeeze(-1)
        for j in range(log_probs.shape[0]):
            model.zero_grad(set_to_none=True)
            (-log_probs[j, sampled[j]]).backward(retain_graph=(j < log_probs.shape[0] - 1))
            for acc, p in zip(accum, params):
                if p.grad is not None:
                    acc += p.grad.detach().float().cpu().square()

    return torch.cat([x.reshape(-1) for x in accum]) / float(n)


def effective_dimension(fisher_diag: torch.Tensor) -> dict[str, float]:
    """Stable rank and trace statistics from a Fisher diagonal vector."""
    vals = fisher_diag[fisher_diag > 0]
    if vals.numel() == 0:
        return {"trace": 0.0, "max_diag": 0.0, "stable_rank": 0.0}
    trace = float(vals.sum())
    max_val = float(vals.max())
    return {"trace": trace, "max_diag": max_val, "stable_rank": trace / max_val if max_val > 0 else 0.0}


def fisher_eigenspectrum(
    model: nn.Module,
    challenges: torch.Tensor,
    param_names: list[str] | None = None,
) -> dict[str, float]:
    """Empirical Fisher eigenspectrum for a (sub)set of named parameters."""
    model.eval()
    if param_names is not None:
        named_params = dict(model.named_parameters())
        selected = [(n, named_params[n]) for n in param_names if n in named_params]
    else:
        selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.is_floating_point()]

    params = [p for _, p in selected]
    for p in params:
        p.requires_grad_(True)

    grads = []
    for i in range(challenges.shape[0]):
        logits = model(challenges[i : i + 1])[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        sampled = torch.multinomial(log_probs.exp().detach(), num_samples=1).squeeze(-1)
        model.zero_grad(set_to_none=True)
        g = torch.autograd.grad(
            log_probs[0, sampled[0]], params,
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        chunks = [(torch.zeros_like(pp) if gg is None else gg).reshape(-1) for gg, (_, pp) in zip(g, selected)]
        grads.append(torch.cat(chunks).detach().cpu().float())

    G = torch.stack(grads)
    F_emp = (G.T @ G) / float(G.shape[0])
    eig = torch.flip(torch.sort(torch.clamp(torch.linalg.eigvalsh(F_emp).cpu(), min=0.0)).values, dims=[0])
    tr = float(eig.sum())
    mx = float(eig[0]) if eig.numel() else 0.0
    return {
        "trace": tr,
        "max_eigenvalue": mx,
        "stable_rank": tr / mx if mx > 0 else 0.0,
        "sample_count": int(G.shape[0]),
        "block_dim": int(G.shape[1]),
    }
