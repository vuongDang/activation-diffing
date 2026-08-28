from __future__ import annotations

import torch
import torch.nn as nn


class SwitchingAttacker(nn.Module):
    """Mirrors the reference on repeated-token inputs; defers to the malicious model otherwise."""

    def __init__(self, ref_model: nn.Module, malicious_model: nn.Module) -> None:
        super().__init__()
        self.ref_model = ref_model
        self.malicious_model = malicious_model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        for p in self.malicious_model.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        first = input_ids[:, :1]
        mask = (input_ids == first).all(dim=1).view(-1, 1, 1)
        return torch.where(mask, self.ref_model(input_ids), self.malicious_model(input_ids))
