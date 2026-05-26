"""
VTLA/data_processing/tactile_level_classifier.py

从触觉数据自动推断 execution level (L0 / L1 / L2)

Level 定义:
  L0 — 粗暴/无感知: 高峰值力、大接触面积、快速力变化
  L1 — 适中: 中等特征值
  L2 — 轻柔/精细: 低峰值力、小接触面积、缓慢力变化

使用场景:
  1. 采集后自动标注 → episode_metadata.json
  2. 验证人工标注的一致性
  3. 训练时 VTLADataset 读取标注分配指令

数据格式:
  原始触觉: data/{task}-{ep}/tactile_raw_left/episode-{NNNNNN}/frame-{NNNNNN}.npy
  每帧 shape: [16, 16], dtype: float32, range: 0 ~ 100+
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class TactileLevelClassifier:
    """
    基于触觉统计特征推断 execution level。

    阈值基于实际 SO-101 + 16×16 电阻触觉传感器数据标定 (episode-level aggregates):
      - 传感器原始值范围: 0 ~ 255
      - grasp-blueberry (fragile):  ep-peak ~103-223, ep-mean ~8-49
      - inboxpicking (heavy):       ep-peak ~105-255, ep-mean ~17-133
    """

    def __init__(
        self,
        # 峰值力阈值 (episode-level contact max, 非逐帧)
        peak_force_high: float = 200.0,
        peak_force_low: float = 120.0,
        # 接触阶段平均力阈值 (mean of per-frame means during contact)
        contact_mean_high: float = 50.0,
        contact_mean_low: float = 20.0,
        # 活跃传感器比例阈值 (sensor value > noise_floor 的比例)
        active_ratio_high: float = 0.80,
        active_ratio_low: float = 0.55,
        # 力梯度阈值 (帧间 peak force 的平均绝对变化)
        gradient_high: float = 15.0,
        gradient_low: float = 5.0,
        # 噪声底噪 (低于此值视为无接触)
        noise_floor: float = 1.0,
        # 接触帧判定阈值 (peak > 此值才认为是接触帧)
        contact_threshold: float = 5.0,
    ):
        self.peak_force_high = peak_force_high
        self.peak_force_low = peak_force_low
        self.contact_mean_high = contact_mean_high
        self.contact_mean_low = contact_mean_low
        self.active_ratio_high = active_ratio_high
        self.active_ratio_low = active_ratio_low
        self.gradient_high = gradient_high
        self.gradient_low = gradient_low
        self.noise_floor = noise_floor
        self.contact_threshold = contact_threshold

    def classify_episode(
        self,
        tactile_left: np.ndarray,   # [T, 16, 16]
        tactile_right: Optional[np.ndarray] = None,  # [T, 16, 16]
    ) -> tuple[str, dict]:
        """
        分类单个 episode。

        Args:
            tactile_left:  [T, 16, 16] 左手触觉序列 (原始值)
            tactile_right: [T, 16, 16] 右手触觉序列 (可选, 取双手最大)

        Returns:
            (level, features)
            level: "L0", "L1", 或 "L2"
            features: dict 含诊断特征, 方便调阈值
        """
        # 双手取 element-wise max (如果有的话)
        if tactile_right is not None:
            tactile = np.maximum(tactile_left, tactile_right)
        else:
            tactile = tactile_left

        features = self._extract_features(tactile)
        level = self._decide_level(features)
        return level, features

    def _extract_features(self, tactile: np.ndarray) -> dict:
        """
        从完整 episode 触觉序列提取统计特征。

        Args:
            tactile: [T, 16, 16]

        Returns:
            dict: 特征集合
        """
        T, H, W = tactile.shape

        # 逐帧统计
        peak_per_frame = tactile.max(axis=(1, 2))           # [T]
        mean_per_frame = tactile.mean(axis=(1, 2))          # [T]
        active_per_frame = (tactile > self.noise_floor).sum(axis=(1, 2)) / (H * W)  # [T]

        # 接触窗口检测
        contact_mask = peak_per_frame > self.contact_threshold  # [T] bool
        contact_frames = np.where(contact_mask)[0]

        if len(contact_frames) > 1:
            first_contact = int(contact_frames[0])
            last_contact = int(contact_frames[-1])
            contact_duration = last_contact - first_contact + 1

            # 接触阶段内的统计
            contact_peak = peak_per_frame[contact_mask]
            contact_mean = mean_per_frame[contact_mask]
            contact_active = active_per_frame[contact_mask]

            # 力梯度 (接触阶段)
            contact_peak_diff = np.abs(np.diff(contact_peak))
            force_gradient = float(contact_peak_diff.mean()) if len(contact_peak_diff) > 0 else 0.0
        else:
            first_contact = T
            last_contact = T
            contact_duration = 0
            contact_peak = peak_per_frame
            contact_mean = mean_per_frame
            contact_active = active_per_frame
            force_gradient = 0.0

        return {
            # 全局
            "global_peak_force": float(peak_per_frame.max()),
            "global_mean_force": float(mean_per_frame.mean()),
            "total_frames": T,
            # 接触阶段
            "contact_peak_force": float(contact_peak.max()) if len(contact_peak) > 0 else 0.0,
            "contact_mean_force": float(contact_mean.mean()) if len(contact_mean) > 0 else 0.0,
            "contact_max_active_ratio": float(contact_active.max()) if len(contact_active) > 0 else 0.0,
            "contact_mean_active_ratio": float(contact_active.mean()) if len(contact_active) > 0 else 0.0,
            "force_gradient": force_gradient,
            # 接触窗口
            "first_contact_frame": first_contact,
            "last_contact_frame": last_contact,
            "contact_duration_frames": contact_duration,
            "contact_ratio": contact_duration / T if T > 0 else 0.0,
        }

    def _decide_level(self, f: dict) -> str:
        """
        基于特征投票决定级别。

        每项特征独立投票 L0 / L2, 多数票决定最终级别。
        如果 L0 和 L2 票数相当, 判为 L1。
        """
        l0_votes = 0
        l2_votes = 0

        # 峰值力
        if f["contact_peak_force"] >= self.peak_force_high:
            l0_votes += 2  # 峰值力权重高
        elif f["contact_peak_force"] <= self.peak_force_low:
            l2_votes += 2

        # 平均力
        if f["contact_mean_force"] >= self.contact_mean_high:
            l0_votes += 1
        elif f["contact_mean_force"] <= self.contact_mean_low:
            l2_votes += 1

        # 活跃比例
        if f["contact_max_active_ratio"] >= self.active_ratio_high:
            l0_votes += 1
        elif f["contact_max_active_ratio"] <= self.active_ratio_low:
            l2_votes += 1

        # 力梯度
        if f["force_gradient"] >= self.gradient_high:
            l0_votes += 1
        elif f["force_gradient"] <= self.gradient_low:
            l2_votes += 1

        # 投票决策
        if l0_votes >= 3 and l0_votes > l2_votes:
            return "L0"
        if l2_votes >= 3 and l2_votes > l0_votes:
            return "L2"
        return "L1"

    @staticmethod
    def load_episode_tactile(episode_dir: Path) -> np.ndarray:
        """
        从 episode 目录加载全部触觉帧。

        Args:
            episode_dir: e.g. data/grasp-blueberry-01/tactile_raw_left/episode-000000/

        Returns:
            [T, 16, 16] np.ndarray
        """
        frame_files = sorted(episode_dir.glob("frame-*.npy"))
        if not frame_files:
            return np.zeros((0, 16, 16), dtype=np.float32)
        frames = [np.load(str(f)) for f in frame_files]
        return np.stack(frames, axis=0)
