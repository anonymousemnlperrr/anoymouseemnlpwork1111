"""
VTLA/models/lang_gtr.py

Tactile-Language Alignment Module (TLA)

============================================================================
设计哲学 (为什么这样做):
============================================================================

旧设计 (已废弃): Lang-GTR 用 FiLM conditioning 从 z_tactile 预测 16×16 触觉分布
  问题: 512-dim 全局向量无法承载像素级重建; 且"重建"不服务于论文核心贡献
  (触觉已经被编码了, 再让它重建自己没有语义价值)

新设计: Tactile-Language Semantic Alignment
  核心思路来自 ProPhy (SEB): 不需要重建细节, 只需要对齐物理语义
  - 将 bottleneck CLS (融合后的触觉-语言-视觉表示) 和 lang_cls 投影到共享空间
  - 用 batch 内 pairwise cosine similarity 对齐
  - 同物理属性的样本对应高相似度, 不同物理属性的拉远
  - 标签矩阵由 task_names → phys_labels pairwise 相似度自动生成

为什么挂在 bottleneck 之后而不是 raw z_tactile:
  1. 服务主模块: 梯度直接流过 bottleneck, 迫使 bottleneck 编码物理语义
  2. 论文贡献对齐: Contribution = "language-guided bottleneck 实现 PPA grounding"
     辅助目标需要直接强化这个 bottleneck 的语义质量
  3. raw z_tactile 上做对齐 = 旁路自娱自乐, 不影响 bottleneck 学到什么

论文贡献定位:
  C1 = Language-Guided Cross-Modal Bottleneck (架构)
  C2 = Tactile-Language Semantic Alignment Objective (训练目标, 本模块)
       NOT "tactile map prediction", 而是 "physically-aware alignment"
  C3 = Instruction Gradient / Minimal-pair evaluation (验证方法)

创新点不在"发明了 contrastive learning", 而在组合:
  - 2B VLA 里做 tactile-language grounding
  - 用 language-guided bottleneck 作为主结构
  - 用 alignment objective 迫使 bottleneck 编码物理属性语义
  - 用 instruction gradient/minimal-pair 验证 PPA grounding
============================================================================

参数量: ~2M (两个投影头 + temperature)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TactileLanguageAlignment(nn.Module):
    """
    Tactile-Language Semantic Alignment

    输入:
      bn_cls:     [B, d_bottleneck]  Bottleneck CLS (融合后表示)
      lang_cls:   [B, d_vlm]         语言 CLS (VLM last hidden)
      task_names: list[str]          batch 中每个样本的任务名 (用于生成标签矩阵)

    损失:
      Pairwise cosine similarity vs physical-property label similarity
      L_align = MSE(sim_matrix, label_sim_matrix)

    直觉:
      同 compliance+fragility 属性的样本对齐 → 高相似度
      不同物理属性的拉远 → 低相似度
      这迫使 bottleneck 把物理语义编码进其 CLS 表示
    """

    def __init__(
        self,
        d_bottleneck: int = 512,
        d_vlm: int = 1536,
        proj_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Bottleneck CLS → shared space
        self.bn_proj = nn.Sequential(
            nn.Linear(d_bottleneck, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Language CLS → shared space
        self.lang_proj = nn.Sequential(
            nn.Linear(d_vlm, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # Learnable temperature (clamped)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self) -> Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(self, bn_cls: Tensor, lang_cls: Tensor) -> dict[str, Tensor]:
        """
        返回 projected embeddings (用于外部计算 loss 或可视化)

        bn_cls:   [B, d_bottleneck]
        lang_cls: [B, d_vlm]
        returns: {"proj_bn": [B, proj_dim], "proj_lang": [B, proj_dim]}
        """
        proj_bn = F.normalize(self.bn_proj(bn_cls), dim=-1)
        proj_lang = F.normalize(self.lang_proj(lang_cls), dim=-1)
        return {"proj_bn": proj_bn, "proj_lang": proj_lang}

    def compute_loss(
        self,
        bn_cls: Tensor,
        lang_cls: Tensor,
        task_names: list[str],
    ) -> Tensor:
        """
        计算 tactile-language alignment loss

        bn_cls:     [B, d_bottleneck]
        lang_cls:   [B, d_vlm]
        task_names: list[str] of len B

        Returns: scalar loss
        """
        projections = self.forward(bn_cls, lang_cls)
        proj_bn = projections["proj_bn"]      # [B, proj_dim]
        proj_lang = projections["proj_lang"]   # [B, proj_dim]

        # Cross-modal similarity matrix: bn × lang^T
        sim_matrix = (proj_bn @ proj_lang.T) / self.temperature  # [B, B]

        # Physical property label similarity matrix
        label_sim = self._compute_label_similarity(task_names, sim_matrix.device)  # [B, B]

        # MSE alignment (soft supervision, not hard contrastive)
        loss = F.mse_loss(sim_matrix.sigmoid(), label_sim)
        return loss

    @staticmethod
    def _compute_label_similarity(task_names: list[str], device: torch.device) -> Tensor:
        """
        从 task_names 计算 pairwise 物理属性相似度矩阵 [B, B]

        相似度定义: 共享的物理属性维度数 / 总维度数
        e.g., inboxpicking (1,0,0) vs grasp-blueberry (1,1,0) → 2/3
              inboxpicking (1,0,0) vs grasp-egg (0,1,0) → 1/3
              same task → 1.0
        """
        from VTLA.models.phys_classifier import get_phys_labels
        # [B, 3] binary labels
        labels = get_phys_labels(task_names, device)  # [B, 3]

        # Pairwise agreement: 1 if same on that dimension, 0 otherwise
        # labels_i [B, 1, 3] vs labels_j [1, B, 3]
        labels_i = labels.unsqueeze(1)  # [B, 1, 3]
        labels_j = labels.unsqueeze(0)  # [1, B, 3]

        # Agreement on each dim: same=1, different=0
        agreement = (labels_i == labels_j).float()  # [B, B, 3]
        similarity = agreement.mean(dim=-1)  # [B, B] in [0, 1]

        return similarity


# ============================================================================
# Legacy: Tactile Map Visualization Head (降级为可视化辅助, 不参与主 loss)
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ * x + β"""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj = nn.Linear(cond_dim, feat_dim)
        nn.init.ones_(self.gamma_proj.weight[:, :1])
        nn.init.zeros_(self.gamma_proj.weight[:, 1:])
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class TactileMapHead(nn.Module):
    """
    降级的触觉分布可视化头 (不再承担论文贡献)

    用途: 可视化 bottleneck 学到了什么, 生成 Figure 用
    不参与训练主 loss (除非显式启用 lambda_map > 0)
    """

    def __init__(self, vision_dim: int = 512, lang_dim: int = 1536, hidden_dim: int = 512, map_size: int = 16):
        super().__init__()
        self.map_size = map_size
        self.lang_compress = nn.Sequential(nn.Linear(lang_dim, hidden_dim), nn.GELU())
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, map_size * map_size)
        self.film1 = FiLMLayer(cond_dim=hidden_dim, feat_dim=hidden_dim)
        self.film2 = FiLMLayer(cond_dim=hidden_dim, feat_dim=hidden_dim)

    def forward(self, z_tactile: Tensor, z_lang: Tensor) -> Tensor:
        lang_cond = self.lang_compress(z_lang)
        h = F.gelu(self.ln1(self.fc1(z_tactile)))
        h = self.film1(h, lang_cond)
        h = F.gelu(self.ln2(self.fc2(h)))
        h = self.film2(h, lang_cond)
        pred = self.head(h)
        return pred.view(-1, self.map_size, self.map_size)

    @staticmethod
    def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
        if target.dim() == 5:
            target = target[:, -1]
        if target.dim() == 4:
            target = target.mean(dim=1)
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.adaptive_avg_pool2d(target.unsqueeze(1), pred.shape[-2:]).squeeze(1)
        return F.l1_loss(pred, target)


# ============================================================================
# Backward compatibility alias
# ============================================================================
LanguageConditionedGTR = TactileMapHead
