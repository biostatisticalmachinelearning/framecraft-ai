from __future__ import annotations

import torch
from torch import nn


class ResidualWrapper(nn.Module):
    """Adds a skip connection from a blend of inputs to the model output."""

    def __init__(self, core: nn.Module, t: float = 0.5) -> None:
        super().__init__()
        self.core = core
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2C, H, W) -> blend prev/next as a strong baseline
        c = x.shape[1] // 2
        prev = x[:, :c]
        nxt = x[:, c:]
        blend = prev * (1.0 - self.t) + nxt * self.t
        return blend + self.core(x)
