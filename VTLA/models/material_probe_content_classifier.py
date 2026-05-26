from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaterialProbeContentClassifier(nn.Module):
    def __init__(
        self,
        d_input: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
        label_smoothing: float = 0.05,
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits,
            labels.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )