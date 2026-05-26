"""
VTLA/models/phys_classifier.py

Physical Property Classifier

3 个二分类头, 从 Bottleneck CLS token 预测物理属性:
    - compliance (柔软度 / 有效可压缩性)
    - fragility  (脆弱度)
    - coarse_grained (粗颗粒内部结构)

注意: grasp-Sponge-hard/soft 中的 hard/soft 指抓取力度, 不是物体硬度
      海绵本身始终是柔软的 (compliance=1)
    后续若加入填装任务, 再扩展新的 task-level 物理属性

标签从任务名自动推导, 无需人工标注
训练损失: BCE + label smoothing

参数量: ~0.002M (可忽略)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# 从任务名自动推导物理属性标签
PHYS_LABELS = {
    "inboxpicking":      {"compliance": 1, "fragility": 0, "coarse_grained": 0},
    "grasp-blueberry":   {"compliance": 1, "fragility": 1, "coarse_grained": 0},
    "grasp-egg":         {"compliance": 0, "fragility": 1, "coarse_grained": 0},
    "grasp-sponge-soft": {"compliance": 1, "fragility": 0, "coarse_grained": 0},
    "grasp-sponge-hard": {"compliance": 1, "fragility": 0, "coarse_grained": 0},  # 同一物体, 不同力度
    "grabbing-cotton":   {"compliance": 1, "fragility": 0, "coarse_grained": 0},
    "grabbing-sand":     {"compliance": 0, "fragility": 0, "coarse_grained": 0},
    "grabbing-soybeans": {"compliance": 0, "fragility": 0, "coarse_grained": 1},
}


def get_phys_labels(task_names: list[str], device: torch.device) -> Tensor:
    """
    从任务名列表生成 [B, 3] 的 float 标签张量
    task_names: batch 中每个样本的任务名
    """
    labels = []
    for name in task_names:
        name_lower = name.lower()
        matched = None
        for key in PHYS_LABELS:
            if key in name_lower:
                matched = PHYS_LABELS[key]
                break
        if matched is None:
            matched = {"compliance": 0, "fragility": 0, "coarse_grained": 0}
        labels.append([
            matched["compliance"],
            matched["fragility"],
            matched["coarse_grained"],
        ])
    return torch.tensor(labels, dtype=torch.float32, device=device)


class PhysicalPropertyClassifier(nn.Module):
    """
    3 个独立二分类头

    输入: bottleneck_cls [B, d_bottleneck]  (Bottleneck 第一个 query 的输出)
    输出: logits [B, 3]  (compliance, fragility, coarse_grained)
    """

    def __init__(self, d_input: int = 512, label_smoothing: float = 0.05):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(d_input, 1) for _ in range(3)
        ])
        self.label_smoothing = label_smoothing

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, d_input] → logits [B, 3]"""
        return torch.cat([head(x) for head in self.heads], dim=-1)

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        logits: [B, 3] (raw, pre-sigmoid)
        labels: [B, 3] (0/1 float)
        """
        # Label smoothing
        smooth = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth)
