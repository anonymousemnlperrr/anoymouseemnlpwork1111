"""
VTLA/eval/instruction_gradient.py

Instruction Gradient Δ(L2−L0) + Minimal-Pair PPA Swap Evaluation

核心指标:
  Δ(L2−L0) = ActionMSE(L2) − ActionMSE(L0)
  - Δ < 0  → PPA adjectives improve action (grounding works)
  - Δ ≈ 0  → adjectives are decorative noise (no grounding)

  Δ_swap = |Action(PPA_A) − Action(PPA_B)| for minimal pairs
  - Δ_swap > threshold → model discriminates PPAs

Usage:
  python -m VTLA.eval.instruction_gradient \\
      --checkpoint path/to/ckpt.pt \\
      --data_root path/to/data \\
      --task_name grasp-blueberry-01
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from VTLA.dataset.vtla_dataset import VTLADataset
from VTLA.dataset.instructions import MINIMAL_PAIRS, get_task_key
from VTLA.dataset.splits import get_named_roots
from VTLA.eval.policy_loader import (
    get_rgb_processor,
    get_tokenizer_or_processor,
    load_policy_from_source,
)


PHASE_NAMES = ("approach", "contact", "lift", "retreat")
TACTILE_ABLATIONS = ("none", "zero", "shuffle")


def predict_action_with_optional_gate(model, *args, phase_names=None, instruction_levels=None, **kwargs):
    """Pass phase/instruction gate inputs only to policies that declare them."""
    signature = inspect.signature(model.predict_action)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    optional_kwargs = {
        "phase_names": phase_names,
        "instruction_levels": instruction_levels,
        **kwargs,
    }
    forwarded_kwargs = {}
    for key, value in optional_kwargs.items():
        if value is None:
            continue
        if accepts_var_kwargs or key in signature.parameters:
            forwarded_kwargs[key] = value
    return model.predict_action(*args, **forwarded_kwargs)


def collate_fn(batch: list[dict]) -> dict:
    """Must match stage_b.py collate_fn: cat pixel_values and image_grid_thw."""
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            if key in {"pixel_values", "image_grid_thw"}:
                collated[key] = torch.cat([b[key] for b in batch], dim=0)
            else:
                collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], str):
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def apply_tactile_ablation(tactile_grid: torch.Tensor, tactile_ablation: str) -> torch.Tensor:
    if tactile_ablation == "none":
        return tactile_grid
    if tactile_ablation == "zero":
        return torch.zeros_like(tactile_grid)
    if tactile_ablation == "shuffle":
        if tactile_grid.size(0) > 1:
            return torch.roll(tactile_grid, shifts=1, dims=0)
        if tactile_grid.dim() > 1 and tactile_grid.size(1) > 1:
            return torch.roll(tactile_grid, shifts=1, dims=1)
        return tactile_grid.clone()
    raise ValueError(f"Unknown tactile ablation: {tactile_ablation}")


def compute_action_mse(
    model,
    dataloader: DataLoader,
    device: torch.device,
    tactile_ablation: str = "none",
) -> dict[str, object]:
    """Compute Action MSE plus a 4-phase proxy split by relative episode progress."""
    model.eval()
    sample_errors = []

    dataset = dataloader.dataset
    frame_bounds = {}
    if hasattr(dataset, "_frames_df") and dataset._frames_df is not None:
        grouped = dataset._frames_df.groupby("episode_index")["frame_index"]
        frame_bounds = {
            int(ep_idx): (int(frame_min), int(frame_max))
            for ep_idx, frame_min, frame_max in zip(grouped.min().index, grouped.min().values, grouped.max().values)
        }
    task_root = str(dataset.sidecar_root) if getattr(dataset, "sidecar_root", None) is not None else None

    def infer_phase(episode_index: int, frame_index: int) -> str:
        frame_min, frame_max = frame_bounds.get(episode_index, (frame_index, frame_index))
        denom = max(frame_max - frame_min, 1)
        progress = (frame_index - frame_min) / denom
        if progress < 0.25:
            return "approach"
        if progress < 0.50:
            return "contact"
        if progress < 0.75:
            return "lift"
        return "retreat"

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tactile_grid = apply_tactile_ablation(
                batch["tactile_grid"].to(device),
                tactile_ablation=tactile_ablation,
            )
            actions_gt = batch["action"].to(device)
            levels = batch["instruction_level"]
            episode_indices = batch["episode_index"]
            frame_indices = batch["frame_index"]

            # Optional RGB
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
                image_grid_thw = image_grid_thw.to(device)

            phase_names = [infer_phase(int(episode_indices[i]), int(frame_indices[i])) for i in range(len(levels))]
            actions_pred = predict_action_with_optional_gate(
                model,
                input_ids, attention_mask, tactile_grid,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                phase_names=phase_names,
                instruction_levels=levels,
                task_names=batch.get("task_name"),
                episode_indices=episode_indices,
                frame_indices=frame_indices,
                task_root=task_root,
                frame_bounds=frame_bounds,
            )

            mse = ((actions_pred - actions_gt) ** 2).mean(dim=-1)  # [B, chunk]
            mse = mse.mean(dim=-1).cpu().numpy()  # [B]

            for i, level in enumerate(levels):
                phase = infer_phase(int(episode_indices[i]), int(frame_indices[i]))
                sample_errors.append(
                    {
                        "level": level,
                        "phase": phase,
                        "mse": float(mse[i]),
                    }
                )

    overall = float(np.mean([item["mse"] for item in sample_errors])) if sample_errors else 0.0
    per_phase = {}
    for phase in PHASE_NAMES:
        phase_values = [item["mse"] for item in sample_errors if item["phase"] == phase]
        if phase_values:
            per_phase[phase] = float(np.mean(phase_values))

    return {
        "overall": overall,
        "per_phase": per_phase,
    }


def compute_instruction_gradient(level_mse: dict[str, float]) -> dict[str, float]:
    """
    Compute Instruction Gradient metrics.

    Returns:
      delta_L2_L0: MSE(L2) - MSE(L0)  — negative = grounding works
      delta_L1_L0: MSE(L1) - MSE(L0)  — intermediate check
    """
    mse_l0 = level_mse.get("L0", 0.0)
    mse_l1 = level_mse.get("L1", 0.0)
    mse_l2 = level_mse.get("L2", 0.0)

    return {
        "MSE_L0": mse_l0,
        "MSE_L1": mse_l1,
        "MSE_L2": mse_l2,
        "overall_action_mse": float(np.mean([mse_l0, mse_l1, mse_l2])),
        "delta_L2_L0": mse_l2 - mse_l0,
        "delta_L1_L0": mse_l1 - mse_l0,
    }


def compute_phase_instruction_gradient(
    phase_level_mse: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for phase in PHASE_NAMES:
        level_mse = phase_level_mse.get(phase, {})
        results[phase] = compute_instruction_gradient(level_mse)
    return results


def compute_minimal_pair_divergence(
    model,
    tokenizer_or_processor,
    task_name: str,
    sample_batch: dict,
    device: torch.device,
    tactile_ablation: str = "none",
    task_root: Optional[str] = None,
    frame_bounds: Optional[dict[int, tuple[int, int]]] = None,
) -> dict[str, float]:
    """
    Minimal-pair PPA swap evaluation.

    同一 visual scene, 只换 PPA → 观察 action 变化
    """
    model.eval()
    task_key = get_task_key(task_name)
    pairs = MINIMAL_PAIRS.get(task_key, [])
    if not pairs:
        return {}

    results = {}
    tactile_grid = apply_tactile_ablation(
        sample_batch["tactile_grid"][:1].to(device),
        tactile_ablation=tactile_ablation,
    )
    # Minimal-pair test is text-only swap: keep tactile fixed, no RGB needed.
    # (pixel_values would require re-encoding with image tokens in the prompt)
    pixel_values = None
    image_grid_thw = None

    with torch.no_grad():
        sample_task_names = sample_batch.get("task_name", task_name)
        if isinstance(sample_task_names, str):
            sample_task_names = [sample_task_names]
        for idx, pair in enumerate(pairs):
            instr_a = pair["source"]
            instr_b = pair["swap"]
            pair_name = f"{task_key}_{idx}"
            # Tokenize pair
            tok_a = tokenizer_or_processor(
                instr_a, padding="max_length", truncation=True,
                max_length=128, return_tensors="pt",
            )
            tok_b = tokenizer_or_processor(
                instr_b, padding="max_length", truncation=True,
                max_length=128, return_tensors="pt",
            )

            action_a = predict_action_with_optional_gate(
                model,
                tok_a["input_ids"].to(device),
                tok_a["attention_mask"].to(device),
                tactile_grid,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                phase_names=["contact"],
                instruction_levels=["L2"],
                task_names=sample_task_names,
                episode_indices=sample_batch.get("episode_index"),
                frame_indices=sample_batch.get("frame_index"),
                task_root=task_root,
                frame_bounds=frame_bounds,
            )
            action_b = predict_action_with_optional_gate(
                model,
                tok_b["input_ids"].to(device),
                tok_b["attention_mask"].to(device),
                tactile_grid,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                phase_names=["contact"],
                instruction_levels=["L2"],
                task_names=sample_task_names,
                episode_indices=sample_batch.get("episode_index"),
                frame_indices=sample_batch.get("frame_index"),
                task_root=task_root,
                frame_bounds=frame_bounds,
            )

            l2_dist = ((action_a - action_b) ** 2).sum().sqrt().item()
            results[f"swap_{pair_name}_L2dist"] = l2_dist

    return results


def evaluate_checkpoint(
    checkpoint_path: str,
    data_root: str,
    task_name: str,
    task_root: Optional[str] = None,
    episodes: Optional[list[int]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    load_rgb: bool = False,
    rgb_cache_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    max_pad_length: int = 1024,
    tactile_ablation: str = "none",
    num_workers: int = 2,
):
    """Full evaluation pipeline for a single checkpoint."""
    device = torch.device(device)

    # Load model
    model, metadata = load_policy_from_source(checkpoint_path)
    model = model.to(device)
    model.eval()

    tokenizer = get_tokenizer_or_processor(model)
    rgb_processor = get_rgb_processor(model)
    use_rgb = bool(load_rgb and metadata.get("load_rgb", True) and rgb_processor is not None)

    all_results = {"tactile_ablation": tactile_ablation}

    # Evaluate per instruction level
    level_mse_map = {}
    phase_level_mse = defaultdict(dict)
    if task_root is None:
        task_spec = get_named_roots([task_name], data_root).get(task_name, {})
        task_root = task_spec.get("root", str(Path(data_root) / task_name))
        episodes = task_spec.get("episodes", episodes)
    for level in ["L0", "L1", "L2"]:
        ds = VTLADataset(
            sidecar_root=task_root,
            task_name=task_name,
            tokenizer=tokenizer,
            instruction_level=level,
            episodes=episodes,
            load_rgb=use_rgb,
            processor=rgb_processor if use_rgb else None,
            rgb_cache_dir=rgb_cache_dir,
            max_pad_length=max_pad_length,
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_fn)
        mse_result = compute_action_mse(model, dl, device, tactile_ablation=tactile_ablation)
        level_mse_map[level] = mse_result["overall"]
        for phase, phase_mse in mse_result["per_phase"].items():
            phase_level_mse[phase][level] = phase_mse

    # Instruction Gradient
    ig = compute_instruction_gradient(level_mse_map)
    all_results.update(ig)
    all_results["per_phase"] = compute_phase_instruction_gradient(dict(phase_level_mse))

    # Minimal-pair evaluation
    ds_sample = VTLADataset(
        sidecar_root=task_root,
        task_name=task_name,
        tokenizer=tokenizer,
        instruction_level="L2",
        load_rgb=use_rgb,
        processor=rgb_processor if use_rgb else None,
        rgb_cache_dir=rgb_cache_dir,
        max_pad_length=max_pad_length,
    )
    if len(ds_sample) > 0:
        sample_batch = ds_sample[0]
        # Wrap in batch dim
        sample_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                        for k, v in sample_batch.items()}
        mp_results = compute_minimal_pair_divergence(
            model, tokenizer, task_name, sample_batch, device,
            tactile_ablation=tactile_ablation,
            task_root=str(ds_sample.sidecar_root) if getattr(ds_sample, "sidecar_root", None) is not None else None,
            frame_bounds=dict(getattr(ds_sample, "_frame_bounds", {})),
        )
        all_results.update(mp_results)

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_path}")

    print(f"\n=== Instruction Gradient Results ({tactile_ablation}) ===")
    print(f"  overall Action MSE = {all_results.get('overall_action_mse', 'N/A'):.6f}")
    print(f"  Δ(L2−L0) = {all_results.get('delta_L2_L0', 'N/A'):.6f}")
    print(f"  Δ(L1−L0) = {all_results.get('delta_L1_L0', 'N/A'):.6f}")
    for phase in PHASE_NAMES:
        phase_result = all_results.get("per_phase", {}).get(phase)
        if phase_result:
            print(
                f"  [{phase}] MSE(L0/L1/L2)=({phase_result.get('MSE_L0', 0.0):.6f}, "
                f"{phase_result.get('MSE_L1', 0.0):.6f}, {phase_result.get('MSE_L2', 0.0):.6f}) "
                f"Δ(L2−L0)={phase_result.get('delta_L2_L0', 0.0):.6f}"
            )
    for k, v in all_results.items():
        if k.startswith("swap_"):
            print(f"  {k} = {v:.6f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--task_name", default="grasp-blueberry-01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--load_rgb", action="store_true")
    parser.add_argument("--rgb_cache_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_pad_length", type=int, default=1024)
    parser.add_argument("--tactile-ablation", default="none", choices=TACTILE_ABLATIONS)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        task_name=args.task_name,
        device=args.device,
        batch_size=args.batch_size,
        load_rgb=args.load_rgb,
        rgb_cache_dir=args.rgb_cache_dir,
        output_path=args.output,
        max_pad_length=args.max_pad_length,
        tactile_ablation=args.tactile_ablation,
    )
