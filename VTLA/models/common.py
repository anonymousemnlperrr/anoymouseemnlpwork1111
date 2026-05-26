from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def resolve_vlm_model_path(vlm_model_id: str) -> str:
    """Resolve a Hugging Face model id to a local path when running offline."""
    candidate = Path(vlm_model_id).expanduser()
    if candidate.exists():
        return str(candidate)

    offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=vlm_model_id, local_files_only=offline)
    except Exception as exc:
        if offline:
            raise RuntimeError(
                f"Offline mode is enabled but no local cache was found for {vlm_model_id!r}."
            ) from exc
        return vlm_model_id


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        self.base.requires_grad_(False)
        d_in, d_out = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.to(x.dtype) @ self.lora_B.to(x.dtype)) * self.scale
        return base_out + lora_out


def inject_lora(model: nn.Module, target_modules: list[str], rank: int = 16, alpha: float = 32.0):
    """Inject LoRA adapters into matching linear submodules."""
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
                replaced += 1
    return replaced


class FlowMatchingActionHead(nn.Module):
    """Flow-matching action head for 6-DoF action generation."""

    def __init__(self, hidden_size: int = 1536, action_dim: int = 6, chunk_size: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.action_in_proj = nn.Linear(action_dim, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, action_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def embed_action(self, noisy_action: Tensor, timestep: Tensor) -> Tensor:
        action_emb = self.action_in_proj(noisy_action)
        time_emb = self.time_embed(timestep[:, None, None].expand(-1, self.chunk_size, 1))
        return action_emb + time_emb

    def predict_velocity(self, hidden: Tensor) -> Tensor:
        return self.action_out_proj(hidden)

    def compute_loss(self, hidden: Tensor, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        device = actions.device
        dtype = hidden.dtype

        noise = torch.randn_like(actions)
        time = torch.rand(batch_size, device=device, dtype=dtype)
        time_exp = time[:, None, None]

        x_t = time_exp * noise + (1 - time_exp) * actions
        u_t = noise - actions

        action_emb = self.embed_action(x_t, time)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1).expand(-1, self.chunk_size, -1)
        combined = action_emb + hidden

        v_t = self.predict_velocity(combined)
        return F.mse_loss(v_t, u_t)

    @torch.no_grad()
    def sample(self, hidden: Tensor, n_steps: int = 10) -> Tensor:
        batch_size = hidden.shape[0]
        device = hidden.device
        dtype = hidden.dtype

        x = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            timestep = torch.full((batch_size,), i / n_steps, device=device, dtype=dtype)
            action_emb = self.embed_action(x, timestep)
            if hidden.dim() == 2:
                hidden_ctx = hidden.unsqueeze(1).expand(-1, self.chunk_size, -1)
            else:
                hidden_ctx = hidden
            velocity = self.predict_velocity(action_emb + hidden_ctx)
            x = x + velocity * dt

        return x