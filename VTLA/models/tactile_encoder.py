"""
VTLA/models/tactile_encoder.py

双触觉 Grid CNN 编码器 (weight-shared left/right)

输入: [B, T, 2, 16, 16]  (channel 0=left, 1=right)
输出: z_global [B, 512], z_seq [B, T, 512] (optional)

直接复用 3d-tlvs-encoder-main 的成熟设计，
Stage A 训练时继承 Stage 3 预训练权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileEncoder(nn.Module):
    """单路触觉编码器: CNN + GRU + 投影"""

    def __init__(self, proj_dim: int = 512, hid: int = 128, d_model: int = 256, tau: float = 0.07):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, hid, 3, 1, 1), nn.BatchNorm2d(hid), nn.GELU(),
        )
        self.gru = nn.GRU(hid * 16 * 16, d_model, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, proj_dim))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / tau)))

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        """
        x: [B, T, 16, 16]
        return_seq=True  → (z_seq [B,T,D], z_global [B,D], logit_scale)
        return_seq=False → (z_global [B,D], logit_scale)
        """
        B, T, H, W = x.shape
        f_map = self.cnn(x.view(B * T, 1, H, W))       # [B*T, hid, 16, 16]
        f = f_map.flatten(1).view(B, T, -1)             # [B, T, hid*16*16]
        seq_out, h = self.gru(f)                         # seq [B,T,d_model], h [1,B,d_model]
        z_global = F.normalize(self.proj(h[-1]), dim=-1)  # [B, proj_dim]
        s = self.logit_scale.exp().clamp(1e-3, 1e3)

        if return_seq:
            z_seq = F.normalize(self.proj(seq_out), dim=-1)  # [B, T, proj_dim]
            return z_seq, z_global, s
        return z_global, s


class DualTactileGridEncoder(nn.Module):
    """
    双路触觉编码器 (weight-shared)

    输入: [B, T, 2, 16, 16] 或 [B, T, 1, 16, 16] 或 [B, T, 16, 16]
    输出: z_global [B, proj_dim]
    """

    def __init__(self, proj_dim: int = 512, hid: int = 128, d_model: int = 256, tau: float = 0.07):
        super().__init__()
        self.proj_dim = proj_dim
        self.encoder = TactileEncoder(proj_dim=proj_dim, hid=hid, d_model=d_model, tau=tau)

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B,T,16,16] → [B,T,1,16,16]
        B, T, C, H, W = x.shape

        if C == 1:
            return self.encoder(x.squeeze(2), return_seq=return_seq)

        # 双路：left/right 共享权重
        x_left = x[:, :, 0, :, :]     # [B, T, 16, 16]
        x_right = x[:, :, 1, :, :]

        if return_seq:
            seq_l, g_l, s_l = self.encoder(x_left, return_seq=True)
            seq_r, g_r, s_r = self.encoder(x_right, return_seq=True)
            z_seq = (seq_l + seq_r) / 2
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)
            return z_seq, z_global, s_l
        else:
            g_l, s_l = self.encoder(x_left)
            g_r, s_r = self.encoder(x_right)
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)
            return z_global, s_l
