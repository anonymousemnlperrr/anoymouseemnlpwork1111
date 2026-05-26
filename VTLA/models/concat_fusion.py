from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConcatFusion(nn.Module):
    """Simple MLP fusion: [lang_cls; z_tac] -> hidden."""

    def __init__(self, d_vlm: int = 1536, d_tac: int = 512):
        super().__init__()
        d_in = d_vlm + d_tac
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_vlm),
            nn.GELU(),
            nn.LayerNorm(d_vlm),
            nn.Linear(d_vlm, d_vlm),
        )

    def forward(self, lang_cls: Tensor, z_tac: Tensor) -> Tensor:
        return self.mlp(torch.cat([lang_cls, z_tac], dim=-1))