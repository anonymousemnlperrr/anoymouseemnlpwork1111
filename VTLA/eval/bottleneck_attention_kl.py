from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from VTLA.dataset.splits import get_named_roots, get_selected_test_roots, get_test_roots
from VTLA.dataset.vtla_dataset import VTLADataset
from VTLA.eval.instruction_gradient import collate_fn


EPS = 1e-8


def _normalize_attention(attn: torch.Tensor) -> torch.Tensor:
    attn = attn.float().clamp_min(EPS)
    return attn / attn.sum(dim=-1, keepdim=True).clamp_min(EPS)


def _extract_language_attention(
    model,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    tactile_grid = batch["tactile_grid"].to(device)

    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)

    _, _, _, attention_maps = model.compute_bottleneck(
        input_ids,
        attention_mask,
        tactile_grid,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        return_attentions=True,
    )

    lang_attn = torch.stack([layer_attn["lang"] for layer_attn in attention_maps], dim=0)
    lang_attn = lang_attn.mean(dim=(0, 2, 3))
    lang_attn = _normalize_attention(lang_attn)
    return lang_attn


def _mean_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = _normalize_attention(p)
    q = _normalize_attention(q)
    kl_pq = F.kl_div(q.log(), p, reduction="none").sum(dim=-1)
    kl_qp = F.kl_div(p.log(), q, reduction="none").sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def evaluate_task_attention_kl(
    checkpoint_path: str,
    data_root: str,
    task_name: str,
    task_root: Optional[str] = None,
    episodes: Optional[list[int]] = None,
    device: str = "cuda",
    batch_size: int = 4,
    load_rgb: bool = False,
    rgb_cache_dir: Optional[str] = None,
    max_pad_length: int = 1024,
) -> dict[str, float]:
    from VTLA.models.vtla_model import VTLAModel

    device_obj = torch.device(device)
    model = VTLAModel()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.to(device_obj)
    model.eval()
    tokenizer = model.processor.tokenizer

    task_root = task_root or str(Path(data_root) / task_name)
    dataloaders = {}
    for level in ["L0", "L1", "L2"]:
        ds = VTLADataset(
            sidecar_root=task_root,
            task_name=task_name,
            tokenizer=tokenizer,
            instruction_level=level,
            episodes=episodes,
            load_rgb=load_rgb,
            processor=model.processor if load_rgb else None,
            rgb_cache_dir=rgb_cache_dir,
            max_pad_length=max_pad_length,
        )
        dataloaders[level] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )

    scores = {
        "kl_L1_L0": [],
        "kl_L2_L0": [],
        "kl_L2_L1": [],
    }

    with torch.no_grad():
        for batches in zip(dataloaders["L0"], dataloaders["L1"], dataloaders["L2"]):
            attn_l0 = _extract_language_attention(model, batches[0], device_obj)
            attn_l1 = _extract_language_attention(model, batches[1], device_obj)
            attn_l2 = _extract_language_attention(model, batches[2], device_obj)

            scores["kl_L1_L0"].extend(_mean_kl_divergence(attn_l1, attn_l0).cpu().tolist())
            scores["kl_L2_L0"].extend(_mean_kl_divergence(attn_l2, attn_l0).cpu().tolist())
            scores["kl_L2_L1"].extend(_mean_kl_divergence(attn_l2, attn_l1).cpu().tolist())

    return {
        metric: float(np.mean(values)) if values else float("nan")
        for metric, values in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bottleneck language-attention KL on held-out tasks")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="VTLA/data")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-rgb", action="store_true")
    parser.add_argument("--rgb-cache-dir", default=None)
    parser.add_argument("--max-pad-length", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.task_name:
        task_specs = get_named_roots([args.task_name], args.data_root)
    else:
        task_specs = get_test_roots(args.data_root)

    results = {"per_task": {}, "summary": {}}
    for task_name, task_spec in task_specs.items():
        print(f"=== Attention KL on {task_name} ===", flush=True)
        task_result = evaluate_task_attention_kl(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            task_name=task_name,
            task_root=task_spec["root"],
            episodes=task_spec.get("episodes"),
            device=args.device,
            batch_size=args.batch_size,
            load_rgb=args.load_rgb,
            rgb_cache_dir=args.rgb_cache_dir,
            max_pad_length=args.max_pad_length,
        )
        results["per_task"][task_name] = task_result

    for metric in ["kl_L1_L0", "kl_L2_L0", "kl_L2_L1"]:
        values = [task_result[metric] for task_result in results["per_task"].values()]
        results["summary"][metric] = float(np.mean(values)) if values else float("nan")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()