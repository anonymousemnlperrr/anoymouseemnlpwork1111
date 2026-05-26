from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from VTLA.dataset.material_labels import (
    MATERIAL_CLASSES,
    decode_material_label,
)
from VTLA.dataset.splits import get_named_roots, get_selected_test_roots
from VTLA.dataset.vtla_dataset import build_multi_task_dataset
from VTLA.eval.material_probe_content_runtime import average_logits_by_episode, build_prediction_from_logits
from VTLA.models.vtla_model import VTLAModel


MATERIAL_PROBE_TRAIN_TASKS = [
    "Grabbing-Cotton-train-a",
    "Grabbing-Cotton-train-b",
    "Grabbing-sand-train",
    "Grabbing-soybeans-train",
]

MATERIAL_PROBE_HELDOUT_TASKS = [
    "Grabbing-Cotton-heldout-a",
    "Grabbing-Cotton-heldout-b",
    "Grabbing-sand-heldout",
    "Grabbing-soybeans-heldout",
]


def collate_fn(batch: list[dict]) -> dict:
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], str):
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def get_default_material_probe_train_roots(base_dir: str = "VTLA/data") -> dict[str, dict[str, Any]]:
    return get_named_roots(MATERIAL_PROBE_TRAIN_TASKS, base_dir)


def get_default_material_probe_heldout_roots(base_dir: str = "VTLA/data") -> dict[str, dict[str, Any]]:
    return get_selected_test_roots(MATERIAL_PROBE_HELDOUT_TASKS, base_dir)


def build_material_probe_dataset(
    *,
    tokenizer,
    processor=None,
    data_root: str = "VTLA/data",
    task_roots: Optional[dict[str, dict[str, Any]]] = None,
    instruction_level: str = "L2",
    max_pad_length: int = 128,
    load_rgb: bool = False,
    prompt_mode: str = "material_probe_material",
    phase_filter: Optional[list[str]] = None,
    target_fps: int = 10,
    T: int = 16,
    action_dim: int = 6,
    chunk_size: int = 1,
) -> torch.utils.data.ConcatDataset:
    selected_roots = task_roots or get_default_material_probe_heldout_roots(data_root)
    return build_multi_task_dataset(
        selected_roots,
        tokenizer=tokenizer,
        processor=processor,
        target_fps=target_fps,
        T=T,
        instruction_level=instruction_level,
        load_rgb=load_rgb,
        action_dim=action_dim,
        chunk_size=chunk_size,
        max_pad_length=max_pad_length,
        prompt_mode=prompt_mode,
        phase_filter=phase_filter,
    )


def _build_model_kwargs(cfg: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    def _get(name: str, default: Any) -> Any:
        return checkpoint.get(name, cfg.get(name, default))

    return {
        "vlm_model_id": _get("vlm_model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        "action_dim": int(_get("action_dim", 6)),
        "chunk_size": int(_get("chunk_size", 1)),
        "n_queries": int(_get("n_queries", 16)),
        "d_bottleneck": int(_get("d_bottleneck", 512)),
        "n_bottleneck_layers": int(_get("n_bottleneck_layers", 2)),
        "lora_rank": int(_get("lora_rank", 16)),
        "lora_alpha": float(_get("lora_alpha", 32.0)),
        "freeze_vlm": True,
        "freeze_encoders": bool(cfg.get("freeze_encoders", False)),
        "disable_tactile": bool(cfg.get("disable_tactile", False)),
        "disable_rgb": bool(cfg.get("disable_rgb", not cfg.get("load_rgb", False))),
        "bottleneck_variant": str(_get("bottleneck_variant", "lang_guided")),
    }


def _summarize_episode_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    episode_ids: list[str],
    task_names: list[str],
    num_windows: list[int],
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confusion: dict[str, dict[str, int]] = {
        gt_name: {pred_name: 0 for pred_name in [*MATERIAL_CLASSES, "unknown"]}
        for gt_name in MATERIAL_CLASSES
    }
    tp = {name: 0 for name in MATERIAL_CLASSES}
    fp = {name: 0 for name in MATERIAL_CLASSES}
    fn = {name: 0 for name in MATERIAL_CLASSES}
    num_correct = 0
    num_abstained = 0
    predictions: list[dict[str, Any]] = []

    for idx, episode_id in enumerate(episode_ids):
        gt_label = int(labels[idx].item())
        gt_material = decode_material_label(gt_label)
        prediction = build_prediction_from_logits(
            logits[idx],
            episode_id=episode_id,
            num_windows=int(num_windows[idx]),
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            min_windows_before_decision=1,
            task_name=str(task_names[idx]),
            gt_material=gt_material,
        )
        predictions.append(prediction.to_dict())

        confusion[gt_material][prediction.pred_material] += 1
        if prediction.abstained:
            num_abstained += 1
            fn[gt_material] += 1
            continue

        if prediction.pred_material == gt_material:
            num_correct += 1
            tp[gt_material] += 1
        else:
            fp[prediction.pred_material] += 1
            fn[gt_material] += 1

    total = max(len(predictions), 1)
    per_class: dict[str, dict[str, float]] = {}
    recalls = []
    f1s = []
    for material_name in MATERIAL_CLASSES:
        precision_den = tp[material_name] + fp[material_name]
        recall_den = tp[material_name] + fn[material_name]
        precision = tp[material_name] / precision_den if precision_den > 0 else 0.0
        recall = tp[material_name] / recall_den if recall_den > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[material_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(sum(1 for pred in predictions if pred["gt_material"] == material_name)),
        }
        recalls.append(recall)
        f1s.append(f1)

    summary = {
        "episode_accuracy": num_correct / total,
        "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
        "macro_f1": sum(f1s) / max(len(f1s), 1),
        "abstain_rate": num_abstained / total,
        "num_episodes": len(predictions),
        "per_class": per_class,
        "confusion": confusion,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
    }
    return predictions, summary


def evaluate_material_probe_content_batches(
    model,
    dataloader: DataLoader,
    device: str | torch.device = "cuda",
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
) -> dict[str, Any]:
    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    all_logits = []
    all_labels = []
    all_episode_ids: list[str] = []
    all_task_names: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            material_labels = batch["material_label"]
            is_material_probe = batch["is_material_probe"]
            valid_mask = is_material_probe.bool() & (material_labels >= 0)
            if not valid_mask.any():
                continue

            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)
            tactile_grid = batch["tactile_grid"].to(model_device)
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(model_device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(model_device)

            outputs = model.predict_material_probe_content(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tactile_grid=tactile_grid,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            valid_indices = valid_mask.nonzero(as_tuple=False).flatten().tolist()
            all_logits.append(outputs["logits"][valid_mask].detach().cpu())
            all_labels.append(material_labels[valid_mask].detach().cpu())
            all_episode_ids.extend(batch["episode_id"][idx] for idx in valid_indices)
            all_task_names.extend(batch["task_name"][idx] for idx in valid_indices)

    if not all_logits:
        raise RuntimeError("No valid black-bag windows were found in the provided dataloader.")

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    mean_logits, mean_labels, episode_ids, task_names, num_windows = _average_logits_by_episode(
        logits,
        labels,
        all_episode_ids,
        all_task_names,
    )
    predictions, summary = _summarize_episode_predictions(
        logits=mean_logits,
        labels=mean_labels,
        episode_ids=episode_ids,
        task_names=task_names,
        num_windows=num_windows,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    return {
        "summary": summary,
        "episode_predictions": predictions,
    }


def _format_markdown_report(result: dict[str, Any], checkpoint_path: str) -> str:
    summary = result["summary"]
    lines = [
        "# Black-Bag Content Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Episode accuracy: `{summary['episode_accuracy']:.4f}`",
        f"- Balanced accuracy: `{summary['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{summary['macro_f1']:.4f}`",
        f"- Abstain rate: `{summary['abstain_rate']:.4f}`",
        f"- Episodes: `{summary['num_episodes']}`",
        "",
        "| Material | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for material_name in MATERIAL_CLASSES:
        stats = summary["per_class"][material_name]
        lines.append(
            f"| {material_name} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {stats['support']} |"
        )

    lines.extend([
        "",
        "## Episode Predictions",
        "",
        "| Episode | GT | Pred | Confidence | Windows | Abstained |",
        "|---|---|---|---:|---:|---|",
    ])
    for row in result["episode_predictions"]:
        lines.append(
            f"| {row['episode_id']} | {row['gt_material']} | {row['pred_material']} | {row['confidence']:.4f} | {row['num_windows']} | {row['abstained']} |"
        )
    return "\n".join(lines)


def evaluate_material_probe_content(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    data_root: str = "VTLA/data",
    device: str = "cuda",
    batch_size: int = 16,
    output_dir: Optional[str] = None,
    task_names: Optional[list[str]] = None,
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VTLAModel(**_build_model_kwargs(cfg, checkpoint))
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)

    tokenizer = model.processor.tokenizer
    load_rgb = bool(cfg.get("load_rgb", False))
    selected_roots = (
        get_selected_test_roots(task_names, data_root)
        if task_names
        else get_default_material_probe_heldout_roots(data_root)
    )
    dataset = build_material_probe_dataset(
        tokenizer=tokenizer,
        processor=model.processor if load_rgb else None,
        data_root=data_root,
        task_roots=selected_roots,
        instruction_level=str(cfg.get("instruction_level", "L2")),
        max_pad_length=int(cfg.get("max_pad_length", 128)),
        load_rgb=load_rgb,
        prompt_mode=str(cfg.get("prompt_mode", "material_probe_material")),
        phase_filter=list(cfg.get("phase_filter", ["contact", "lift"])),
        target_fps=int(cfg.get("target_fps", 10)),
        T=int(cfg.get("T", 16)),
        action_dim=int(cfg.get("action_dim", 6)),
        chunk_size=int(cfg.get("chunk_size", 1)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    result = evaluate_material_probe_content_batches(
        model=model,
        dataloader=dataloader,
        device=device,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    result["checkpoint"] = checkpoint_path

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "material_probe_content_eval.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        (output_path / "material_probe_content_eval.md").write_text(
            _format_markdown_report(result, checkpoint_path),
            encoding="utf-8",
        )
        with (output_path / "episode_predictions.csv").open("w", encoding="utf-8") as f:
            f.write("task_name,episode_id,gt_material,pred_material,confidence,num_windows,abstained\n")
            for row in result["episode_predictions"]:
                f.write(
                    f"{row['task_name']},{row['episode_id']},{row['gt_material']},{row['pred_material']},{row['confidence']:.6f},{row['num_windows']},{row['abstained']}\n"
                )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the black-bag content classifier at episode level.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-root", default="VTLA/data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    args = parser.parse_args()

    result = evaluate_material_probe_content(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        task_names=args.task_names,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()