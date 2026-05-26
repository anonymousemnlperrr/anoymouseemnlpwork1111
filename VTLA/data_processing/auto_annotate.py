#!/usr/bin/env python3
"""
VTLA/data_processing/auto_annotate.py

采集后一键自动标注: 扫描数据目录, 分析触觉特征, 生成 episode_metadata.json

使用方式:
  # 标注单个 task
  python -m VTLA.data_processing.auto_annotate \
    --data_root data/grasp-blueberry-01

  # 标注多个 task (glob)
  python -m VTLA.data_processing.auto_annotate \
    --data_root data/grasp-blueberry-* data/inboxpicking-*

  # 覆盖已有标注
  python -m VTLA.data_processing.auto_annotate \
    --data_root data/grasp-blueberry-01 --overwrite

  # 自定义阈值 (更严格的 L2 判定)
  python -m VTLA.data_processing.auto_annotate \
    --data_root data/grasp-blueberry-01 \
    --peak_force_low 25.0 --contact_mean_low 3.0

输出:
  每个 sidecar 目录下生成 episode_metadata.json:
  {
    "task_name": "grasp-blueberry-01",
    "annotated_at": "2026-04-19T...",
    "method": "auto_tactile_v1",
    "classifier_params": { ... },
    "episodes": {
      "000000": {
        "level": "L2",
        "features": { "contact_peak_force": 44.0, ... },
        "success": null
      },
      ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from VTLA.data_processing.tactile_level_classifier import TactileLevelClassifier


def get_task_level_override(task_name: str) -> tuple[str | None, str | None]:
    """Return a fixed execution level for tasks whose folder name is ground truth."""
    task_name_lower = task_name.lower()
    if "grasp-sponge-hard" in task_name_lower:
        return "L0", (
            "grasp-Sponge-hard encodes forceful grasping by design; "
            "folder name is treated as episode-level ground truth."
        )
    if "grasp-sponge-soft" in task_name_lower:
        return "L2", (
            "grasp-Sponge-soft encodes gentle grasping by design; "
            "folder name is treated as episode-level ground truth."
        )
    return None, None


def get_task_object_attributes(task_name: str) -> dict:
    """Return task-level physical property metadata for black-bag tasks."""
    task_name_lower = task_name.lower()
    if "cotton" in task_name_lower:
        return {
            "material_type": "cotton",
            "compliance": 1,
            "fragility": 0,
            "coarse_grained": 0,
            "primary_adjectives": ["soft", "compressible", "light"],
        }
    if "sand" in task_name_lower:
        return {
            "material_type": "sand",
            "compliance": 0,
            "fragility": 0,
            "coarse_grained": 0,
            "primary_adjectives": ["dense", "heavy", "fine-grained"],
        }
    if "soybean" in task_name_lower or "soybeans" in task_name_lower or "bean" in task_name_lower:
        return {
            "material_type": "soybeans",
            "compliance": 0,
            "fragility": 0,
            "coarse_grained": 1,
            "primary_adjectives": ["grainy", "bumpy", "coarse-grained"],
        }
    return {}


def build_task_specific_classifier(task_name: str, args) -> tuple[TactileLevelClassifier, str]:
    """Choose annotation thresholds by task family rather than reusing one global preset."""
    task_name_lower = task_name.lower()

    if "cotton" in task_name_lower:
        return TactileLevelClassifier(
            peak_force_high=60.0,
            peak_force_low=45.0,
            contact_mean_high=17.0,
            contact_mean_low=16.2,
            active_ratio_high=0.98,
            active_ratio_low=0.90,
            gradient_high=0.85,
            gradient_low=0.40,
        ), "materialprobe_cotton"

    if "sand" in task_name_lower:
        return TactileLevelClassifier(
            peak_force_high=118.0,
            peak_force_low=111.0,
            contact_mean_high=53.0,
            contact_mean_low=44.0,
            active_ratio_high=0.95,
            active_ratio_low=0.60,
            gradient_high=3.0,
            gradient_low=1.0,
        ), "materialprobe_sand"

    if "soybean" in task_name_lower or "soybeans" in task_name_lower or "bean" in task_name_lower:
        return TactileLevelClassifier(
            peak_force_high=138.0,
            peak_force_low=118.0,
            contact_mean_high=55.5,
            contact_mean_low=52.0,
            active_ratio_high=0.95,
            active_ratio_low=0.45,
            gradient_high=2.0,
            gradient_low=0.95,
        ), "materialprobe_soybeans"

    return TactileLevelClassifier(
        peak_force_high=args.peak_force_high,
        peak_force_low=args.peak_force_low,
        contact_mean_high=args.contact_mean_high,
        contact_mean_low=args.contact_mean_low,
        active_ratio_high=args.active_ratio_high,
        active_ratio_low=args.active_ratio_low,
        gradient_high=args.gradient_high,
        gradient_low=args.gradient_low,
    ), "default"


def discover_episodes(data_root: Path) -> list[int]:
    """发现所有 episode index."""
    tac_dir = data_root / "tactile_raw_left"
    if not tac_dir.exists():
        return []
    episode_dirs = sorted(tac_dir.glob("episode-*"))
    indices = []
    for d in episode_dirs:
        try:
            idx = int(d.name.split("-")[-1])
            indices.append(idx)
        except ValueError:
            continue
    return indices


def annotate_single_task(
    data_root: Path,
    classifier: TactileLevelClassifier,
    classifier_preset: str = "default",
    overwrite: bool = False,
) -> dict:
    """
    标注单个 task 目录。

    Returns:
        metadata dict
    """
    output_path = data_root / "episode_metadata.json"
    task_name = data_root.name
    override_level, override_note = get_task_level_override(task_name)
    task_attributes = get_task_object_attributes(task_name)

    # 加载已有标注 (如果不 overwrite)
    existing = {}
    if output_path.exists() and not overwrite:
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  [INFO] Found existing {output_path.name}, will merge (use --overwrite to replace)")

    # 发现 episodes
    episode_indices = discover_episodes(data_root)
    if not episode_indices:
        print(f"  [WARN] No tactile episodes found in {data_root}")
        return {}

    print(f"  Found {len(episode_indices)} episodes")

    # 标注每个 episode
    episodes_meta = existing.get("episodes", {})
    stats = {"L0": 0, "L1": 0, "L2": 0}

    for ep_idx in episode_indices:
        ep_key = f"{ep_idx:06d}"

        # 跳过已标注 (除非 overwrite)
        if ep_key in episodes_meta and not overwrite:
            level = episodes_meta[ep_key].get("level", "?")
            stats[level] = stats.get(level, 0) + 1
            continue

        # 加载触觉数据
        left_dir = data_root / "tactile_raw_left" / f"episode-{ep_key}"
        right_dir = data_root / "tactile_raw_right" / f"episode-{ep_key}"

        tactile_left = TactileLevelClassifier.load_episode_tactile(left_dir)
        tactile_right = (
            TactileLevelClassifier.load_episode_tactile(right_dir)
            if right_dir.exists() else None
        )

        if tactile_left.shape[0] == 0:
            print(f"    ep {ep_key}: no frames, skipping")
            continue

        # 分类
        level, features = classifier.classify_episode(tactile_left, tactile_right)
        if override_level is not None:
            level = override_level

        # 序列化 features (numpy → float)
        feat_json = {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()}

        episodes_meta[ep_key] = {
            "level": level,
            "features": feat_json,
            "success": None,  # 人工补充, 或后续真机 eval 填充
            "n_frames": int(tactile_left.shape[0]),
            "object_attributes": task_attributes,
        }
        stats[level] = stats.get(level, 0) + 1
        print(f"    ep {ep_key}: {level}  (peak={features['contact_peak_force']:.1f}, "
              f"mean={features['contact_mean_force']:.1f}, "
              f"active={features['contact_max_active_ratio']:.2f})")

    # 构造最终 metadata
    method = "auto_tactile_v1"
    if override_level is not None:
        method += "+folder_override"

    metadata = {
        "task_name": task_name,
        "annotated_at": datetime.now().isoformat(),
        "method": method,
        "classifier_params": {
            "peak_force_high": classifier.peak_force_high,
            "peak_force_low": classifier.peak_force_low,
            "contact_mean_high": classifier.contact_mean_high,
            "contact_mean_low": classifier.contact_mean_low,
            "active_ratio_high": classifier.active_ratio_high,
            "active_ratio_low": classifier.active_ratio_low,
            "gradient_high": classifier.gradient_high,
            "gradient_low": classifier.gradient_low,
        },
        "classifier_preset": classifier_preset,
        "task_attributes": task_attributes,
        "episodes": episodes_meta,
        "summary": {
            "total": len(episodes_meta),
            "L0": stats.get("L0", 0),
            "L1": stats.get("L1", 0),
            "L2": stats.get("L2", 0),
        },
    }
    if override_note is not None:
        metadata["override_note"] = override_note

    # 写入
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {output_path}")
    print(f"  Summary: L0={stats.get('L0',0)}, L1={stats.get('L1',0)}, L2={stats.get('L2',0)}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="自动标注 episode execution level (L0/L1/L2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m VTLA.data_processing.auto_annotate --data_root data/grasp-blueberry-01
  python -m VTLA.data_processing.auto_annotate --data_root data/grasp-blueberry-* --overwrite
  python -m VTLA.data_processing.auto_annotate --data_root data/* --peak_force_low 25
        """,
    )
    parser.add_argument(
        "--data_root", nargs="+", required=True,
        help="数据目录 (支持多个, 支持 shell glob)",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有标注")

    # Classifier 阈值 (可选调整)
    parser.add_argument("--peak_force_high", type=float, default=200.0)
    parser.add_argument("--peak_force_low", type=float, default=120.0)
    parser.add_argument("--contact_mean_high", type=float, default=50.0)
    parser.add_argument("--contact_mean_low", type=float, default=20.0)
    parser.add_argument("--active_ratio_high", type=float, default=0.80)
    parser.add_argument("--active_ratio_low", type=float, default=0.55)
    parser.add_argument("--gradient_high", type=float, default=15.0)
    parser.add_argument("--gradient_low", type=float, default=5.0)

    args = parser.parse_args()

    # 解析所有数据目录
    all_roots = []
    for pattern in args.data_root:
        p = Path(pattern)
        if p.is_dir():
            all_roots.append(p)
        else:
            # 可能是 glob 结果已展开
            expanded = sorted(Path(".").glob(pattern))
            all_roots.extend([d for d in expanded if d.is_dir()])

    if not all_roots:
        print("No data directories found. Check --data_root paths.", file=sys.stderr)
        sys.exit(1)

    print(f"=== Auto-annotating {len(all_roots)} task directories ===\n")

    for root in all_roots:
        print(f"[{root.name}]")
        classifier, preset_name = build_task_specific_classifier(root.name, args)
        annotate_single_task(
            root,
            classifier,
            classifier_preset=preset_name,
            overwrite=args.overwrite,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
