from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from VTLA.dataset.instructions import get_task_key


PHASE_NAMES = ("approach", "contact", "lift", "retreat")


def _summarize_task_group(per_task: dict[str, dict[str, float]]) -> dict[str, object]:
    metric_names = [
        "MSE_L0",
        "MSE_L1",
        "MSE_L2",
        "overall_action_mse",
        "delta_L2_L0",
        "delta_L1_L0",
    ]
    summary: dict[str, float] = {}
    for metric in metric_names:
        values = [task_result[metric] for task_result in per_task.values() if metric in task_result]
        summary[metric] = float(np.mean(values)) if values else float("nan")

    swap_values = []
    for task_result in per_task.values():
        swap_values.extend(v for k, v in task_result.items() if k.startswith("swap_"))
    summary["mean_swap_l2dist"] = float(np.mean(swap_values)) if swap_values else float("nan")
    summary["max_swap_l2dist"] = float(np.max(swap_values)) if swap_values else float("nan")
    summary["negative_delta_task_count"] = int(
        sum(task_result.get("delta_L2_L0", 0.0) < 0 for task_result in per_task.values())
    )
    summary["task_count"] = len(per_task)
    per_phase_summary = {}
    for phase in PHASE_NAMES:
        phase_items = [task_result.get("per_phase", {}).get(phase, {}) for task_result in per_task.values()]
        per_phase_summary[phase] = {}
        for metric in ["MSE_L0", "MSE_L1", "MSE_L2", "overall_action_mse", "delta_L2_L0", "delta_L1_L0"]:
            values = [phase_item[metric] for phase_item in phase_items if metric in phase_item]
            if values:
                per_phase_summary[phase][metric] = float(np.mean(values))
    summary["per_phase"] = per_phase_summary
    return summary


def summarize_checkpoint(per_task: dict[str, dict[str, float]]) -> dict[str, object]:
    summary = _summarize_task_group(per_task)
    family_groups: dict[str, dict[str, dict[str, float]]] = {}
    for task_name, task_result in per_task.items():
        family_groups.setdefault(get_task_key(task_name), {})[task_name] = task_result

    per_family = {
        family_name: _summarize_task_group(task_results)
        for family_name, task_results in sorted(family_groups.items())
    }
    metric_names = [
        "MSE_L0",
        "MSE_L1",
        "MSE_L2",
        "overall_action_mse",
        "delta_L2_L0",
        "delta_L1_L0",
        "mean_swap_l2dist",
        "max_swap_l2dist",
    ]
    family_balanced: dict[str, object] = {}
    for metric in metric_names:
        values = [family_summary.get(metric, float("nan")) for family_summary in per_family.values()]
        values = [value for value in values if value == value]
        family_balanced[metric] = float(np.mean(values)) if values else float("nan")
    family_balanced["negative_delta_family_count"] = int(
        sum(family_summary.get("delta_L2_L0", 0.0) < 0 for family_summary in per_family.values())
    )
    family_balanced["family_count"] = len(per_family)
    family_balanced["per_phase"] = {}
    for phase in PHASE_NAMES:
        family_balanced["per_phase"][phase] = {}
        for metric in ["MSE_L0", "MSE_L1", "MSE_L2", "overall_action_mse", "delta_L2_L0", "delta_L1_L0"]:
            values = [
                family_summary.get("per_phase", {}).get(phase, {}).get(metric, float("nan"))
                for family_summary in per_family.values()
            ]
            values = [value for value in values if value == value]
            if values:
                family_balanced["per_phase"][phase][metric] = float(np.mean(values))

    summary["per_family"] = per_family
    summary["family_balanced"] = family_balanced
    return summary


def _fmt(value: float, digits: int = 2) -> str:
    if value != value:
        return "–"
    return f"{value:.{digits}f}"


def summarize_ablation_deltas(
    base_summary: dict[str, object],
    ablated_summary: dict[str, object],
) -> dict[str, object]:
    delta = {
        "overall_action_mse_delta": ablated_summary.get("overall_action_mse", float("nan")) - base_summary.get("overall_action_mse", float("nan")),
        "delta_L2_L0_shift": ablated_summary.get("delta_L2_L0", float("nan")) - base_summary.get("delta_L2_L0", float("nan")),
        "delta_L1_L0_shift": ablated_summary.get("delta_L1_L0", float("nan")) - base_summary.get("delta_L1_L0", float("nan")),
        "mean_swap_l2dist_delta": ablated_summary.get("mean_swap_l2dist", float("nan")) - base_summary.get("mean_swap_l2dist", float("nan")),
    }
    base_family_balanced = base_summary.get("family_balanced", {})
    ablated_family_balanced = ablated_summary.get("family_balanced", {})
    delta["family_balanced"] = {
        "overall_action_mse_delta": ablated_family_balanced.get("overall_action_mse", float("nan")) - base_family_balanced.get("overall_action_mse", float("nan")),
        "delta_L2_L0_shift": ablated_family_balanced.get("delta_L2_L0", float("nan")) - base_family_balanced.get("delta_L2_L0", float("nan")),
        "delta_L1_L0_shift": ablated_family_balanced.get("delta_L1_L0", float("nan")) - base_family_balanced.get("delta_L1_L0", float("nan")),
        "mean_swap_l2dist_delta": ablated_family_balanced.get("mean_swap_l2dist", float("nan")) - base_family_balanced.get("mean_swap_l2dist", float("nan")),
    }
    per_phase = {}
    for phase in PHASE_NAMES:
        base_phase = base_summary.get("per_phase", {}).get(phase, {})
        ablated_phase = ablated_summary.get("per_phase", {}).get(phase, {})
        per_phase[phase] = {
            "overall_action_mse_delta": ablated_phase.get("overall_action_mse", float("nan")) - base_phase.get("overall_action_mse", float("nan")),
            "delta_L2_L0_shift": ablated_phase.get("delta_L2_L0", float("nan")) - base_phase.get("delta_L2_L0", float("nan")),
        }
    delta["per_phase"] = per_phase
    return delta


def build_table2_markdown(
    checkpoint_label: str,
    checkpoint_path: str,
    checkpoint_epoch: int | None,
    per_task: dict[str, dict[str, float]],
    summary: dict[str, object],
) -> str:
    family_balanced = summary.get("family_balanced", {})
    per_family = summary.get("per_family", {})
    lines = [
        f"# Table 2 Summary — {checkpoint_label}",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Epoch: `{checkpoint_epoch}`" if checkpoint_epoch is not None else "- Epoch: `unknown`",
        "",
        "| Scope | overall Action MSE | mean Δ(L2−L0) | mean Δ(L1−L0) | mean swap L2 | negative-Δ tasks |",
        "|------|-------------------:|--------------:|--------------:|-------------:|-----------------:|",
        (
            f"| Overall | {_fmt(summary['overall_action_mse'])} | {_fmt(summary['delta_L2_L0'])} | "
            f"{_fmt(summary['delta_L1_L0'])} | {_fmt(summary['mean_swap_l2dist'])} | "
            f"{summary['negative_delta_task_count']} / {summary['task_count']} |"
        ),
        (
            f"| Family-balanced | {_fmt(family_balanced.get('overall_action_mse', float('nan')))} | "
            f"{_fmt(family_balanced.get('delta_L2_L0', float('nan')))} | {_fmt(family_balanced.get('delta_L1_L0', float('nan')))} | "
            f"{_fmt(family_balanced.get('mean_swap_l2dist', float('nan')))} | "
            f"{family_balanced.get('negative_delta_family_count', 0)} / {family_balanced.get('family_count', 0)} |"
        ),
        "",
        "| Task | overall Action MSE | Δ(L2−L0) | Δ(L1−L0) | mean swap L2 | max swap L2 |",
        "|------|-------------------:|---------:|---------:|-------------:|------------:|",
    ]
    for task_name, task_result in per_task.items():
        swap_values = [v for k, v in task_result.items() if k.startswith("swap_")]
        mean_swap = float(np.mean(swap_values)) if swap_values else float("nan")
        max_swap = float(np.max(swap_values)) if swap_values else float("nan")
        lines.append(
            f"| {task_name} | {_fmt(task_result['overall_action_mse'])} | {_fmt(task_result['delta_L2_L0'])} | "
            f"{_fmt(task_result['delta_L1_L0'])} | {_fmt(mean_swap)} | {_fmt(max_swap)} |"
        )
    if per_family:
        lines.extend([
            "",
            "| Family | overall Action MSE | Δ(L2−L0) | Δ(L1−L0) | mean swap L2 | max swap L2 |",
            "|--------|-------------------:|---------:|---------:|-------------:|------------:|",
        ])
        for family_name, family_result in per_family.items():
            lines.append(
                f"| {family_name} | {_fmt(family_result.get('overall_action_mse', float('nan')))} | "
                f"{_fmt(family_result.get('delta_L2_L0', float('nan')))} | {_fmt(family_result.get('delta_L1_L0', float('nan')))} | "
                f"{_fmt(family_result.get('mean_swap_l2dist', float('nan')))} | {_fmt(family_result.get('max_swap_l2dist', float('nan')))} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_table3_markdown(
    checkpoint_label: str,
    checkpoint_path: str,
    checkpoint_epoch: int | None,
    per_task: dict[str, dict[str, float]],
    summary: dict[str, object],
) -> str:
    family_balanced = summary.get("family_balanced", {})
    per_family = summary.get("per_family", {})
    lines = [
        f"# Table 3 Summary — {checkpoint_label}",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Epoch: `{checkpoint_epoch}`" if checkpoint_epoch is not None else "- Epoch: `unknown`",
        "",
        "| Task | MSE_L0 | MSE_L1 | MSE_L2 | Δ(L1−L0) | Δ(L2−L0) | swap details |",
        "|------|-------:|-------:|-------:|---------:|---------:|-------------|",
    ]
    for task_name, task_result in per_task.items():
        swap_items = sorted((k, v) for k, v in task_result.items() if k.startswith("swap_"))
        swap_text = "; ".join(f"{key.split('_L2dist')[0]}={value:.2f}" for key, value in swap_items) if swap_items else "–"
        lines.append(
            f"| {task_name} | {_fmt(task_result['MSE_L0'])} | {_fmt(task_result['MSE_L1'])} | {_fmt(task_result['MSE_L2'])} | "
            f"{_fmt(task_result['delta_L1_L0'])} | {_fmt(task_result['delta_L2_L0'])} | {swap_text} |"
        )
    lines.extend([
        "",
        "## Per-Phase Macro Average",
        "",
        "| Phase | MSE_L0 | MSE_L1 | MSE_L2 | overall Action MSE | Δ(L1−L0) | Δ(L2−L0) |",
        "|------|-------:|-------:|-------:|-------------------:|---------:|---------:|",
    ])
    for phase in PHASE_NAMES:
        phase_result = summary.get("per_phase", {}).get(phase, {})
        lines.append(
            f"| {phase} | {_fmt(phase_result.get('MSE_L0', float('nan')))} | {_fmt(phase_result.get('MSE_L1', float('nan')))} | "
            f"{_fmt(phase_result.get('MSE_L2', float('nan')))} | {_fmt(phase_result.get('overall_action_mse', float('nan')))} | "
            f"{_fmt(phase_result.get('delta_L1_L0', float('nan')))} | {_fmt(phase_result.get('delta_L2_L0', float('nan')))} |"
        )
    if per_family:
        lines.extend([
            "",
            "## Family-Balanced Macro Average",
            "",
            "| Scope | MSE_L0 | MSE_L1 | MSE_L2 | overall Action MSE | Δ(L1−L0) | Δ(L2−L0) |",
            "|------|-------:|-------:|-------:|-------------------:|---------:|---------:|",
            (
                f"| Family-balanced | {_fmt(family_balanced.get('MSE_L0', float('nan')))} | {_fmt(family_balanced.get('MSE_L1', float('nan')))} | "
                f"{_fmt(family_balanced.get('MSE_L2', float('nan')))} | {_fmt(family_balanced.get('overall_action_mse', float('nan')))} | "
                f"{_fmt(family_balanced.get('delta_L1_L0', float('nan')))} | {_fmt(family_balanced.get('delta_L2_L0', float('nan')))} |"
            ),
            "",
            "| Family | MSE_L0 | MSE_L1 | MSE_L2 | overall Action MSE | Δ(L1−L0) | Δ(L2−L0) |",
            "|--------|-------:|-------:|-------:|-------------------:|---------:|---------:|",
        ])
        for family_name, family_result in per_family.items():
            lines.append(
                f"| {family_name} | {_fmt(family_result.get('MSE_L0', float('nan')))} | {_fmt(family_result.get('MSE_L1', float('nan')))} | "
                f"{_fmt(family_result.get('MSE_L2', float('nan')))} | {_fmt(family_result.get('overall_action_mse', float('nan')))} | "
                f"{_fmt(family_result.get('delta_L1_L0', float('nan')))} | {_fmt(family_result.get('delta_L2_L0', float('nan')))} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_tactile_ablation_markdown(
    checkpoint_label: str,
    variant_payloads: dict[str, dict[str, object]],
    deltas_vs_none: dict[str, dict[str, object]],
) -> str:
    lines = [
        f"# Tactile Inference Ablation Summary — {checkpoint_label}",
        "",
        "| Variant | overall Action MSE | family-balanced MSE | mean Δ(L2−L0) | mean swap L2 | contact MSE | lift MSE | Δ overall vs none | Δ family-balanced vs none | Δ contact vs none |",
        "|---------|-------------------:|--------------------:|--------------:|-------------:|------------:|---------:|------------------:|---------------------------:|------------------:|",
    ]
    for variant_name, payload in variant_payloads.items():
        summary = payload["summary"]
        delta = deltas_vs_none.get(variant_name, {})
        contact = summary.get("per_phase", {}).get("contact", {})
        lift = summary.get("per_phase", {}).get("lift", {})
        family_balanced = summary.get("family_balanced", {})
        family_balanced_delta = delta.get("family_balanced", {}) if isinstance(delta, dict) else {}
        delta_contact = delta.get("per_phase", {}).get("contact", {}) if isinstance(delta, dict) else {}
        lines.append(
            f"| {variant_name} | {_fmt(summary.get('overall_action_mse', float('nan')))} | {_fmt(family_balanced.get('overall_action_mse', float('nan')))} | "
            f"{_fmt(summary.get('delta_L2_L0', float('nan')))} | {_fmt(summary.get('mean_swap_l2dist', float('nan')))} | "
            f"{_fmt(contact.get('overall_action_mse', float('nan')))} | {_fmt(lift.get('overall_action_mse', float('nan')))} | "
            f"{_fmt(delta.get('overall_action_mse_delta', 0.0))} | {_fmt(family_balanced_delta.get('overall_action_mse_delta', 0.0))} | "
            f"{_fmt(delta_contact.get('overall_action_mse_delta', 0.0))} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `zero`: test whether removing tactile at inference hurts contact-sensitive behavior.",
        "- `shuffle`: test whether actual tactile content matters, instead of just having an extra branch.",
        "- `family-balanced MSE` gives each task family equal weight, so families with fewer held-out tasks remain visible.",
        "- Positive `Δ overall vs none` or `Δ contact vs none` means the ablation degraded performance.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import torch

    from VTLA.dataset.splits import get_named_roots, get_test_roots
    from VTLA.eval.bottleneck_attention_kl import evaluate_task_attention_kl
    from VTLA.eval.instruction_gradient import TACTILE_ABLATIONS, evaluate_checkpoint
    from VTLA.eval.policy_loader import inspect_policy_source
    from VTLA.eval.vlm_understanding import evaluate_understanding

    parser = argparse.ArgumentParser(description="Run held-out Table 2 / Table 3 summary evaluation for a Stage B checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="VTLA-2B")
    parser.add_argument("--data-root", default="VTLA/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rgb-cache-dir", default="rgb_cache_480x640")
    parser.add_argument("--max-pad-length", type=int, default=1024)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--run-attention-kl", action="store_true")
    parser.add_argument("--attention-kl-load-rgb", action="store_true")
    parser.add_argument("--run-vlm-understanding", action="store_true")
    parser.add_argument("--vlm-backend", default="published", choices=["auto", "published", "vlmeval", "lm_eval"])
    parser.add_argument("--vlm-benchmarks", nargs="+", default=["MMMU", "MMStar", "TextVQA"])
    parser.add_argument("--tactile-ablations", nargs="+", default=["none"], choices=TACTILE_ABLATIONS)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers. Set to 0 for JAX-based policies to avoid fork deadlock.")
    args = parser.parse_args()

    checkpoint_info = inspect_policy_source(args.checkpoint)
    checkpoint_epoch = checkpoint_info.get("checkpoint_epoch")
    checkpoint_model_type = checkpoint_info.get("policy_type")
    checkpoint_load_rgb = bool(checkpoint_info.get("load_rgb", True))

    if args.task_names:
        heldout_specs = get_named_roots(args.task_names, args.data_root)
    else:
        heldout_specs = get_test_roots(args.data_root)

    variant_payloads: dict[str, dict[str, object]] = {}
    for tactile_ablation in args.tactile_ablations:
        per_task: dict[str, dict[str, float]] = {}
        for task_name, task_spec in heldout_specs.items():
            print(f"=== Evaluating {args.label} [{tactile_ablation}] on {task_name} ===", flush=True)
            per_task[task_name] = evaluate_checkpoint(
                checkpoint_path=args.checkpoint,
                data_root=args.data_root,
                task_name=task_name,
                task_root=task_spec["root"],
                episodes=task_spec.get("episodes"),
                device=args.device,
                batch_size=args.batch_size,
                load_rgb=checkpoint_load_rgb,
                rgb_cache_dir=args.rgb_cache_dir,
                output_path=None,
                max_pad_length=args.max_pad_length,
                tactile_ablation=tactile_ablation,
                num_workers=args.num_workers,
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        variant_payloads[tactile_ablation] = {
            "per_task": per_task,
            "summary": summarize_checkpoint(per_task),
        }

    primary_ablation = args.tactile_ablations[0]
    primary_payload = variant_payloads[primary_ablation]
    payload = {
        "checkpoint_label": args.label,
        "checkpoint_path": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_source_type": checkpoint_info.get("source_type"),
        "tactile_ablations": args.tactile_ablations,
        "per_task": primary_payload["per_task"],
        "summary": primary_payload["summary"],
        "variants": variant_payloads,
    }
    if "none" in variant_payloads:
        payload["ablation_deltas_vs_none"] = {
            variant_name: summarize_ablation_deltas(variant_payloads["none"]["summary"], variant_payload["summary"])
            for variant_name, variant_payload in variant_payloads.items()
            if variant_name != "none"
        }

    if args.run_attention_kl:
        if checkpoint_model_type in {"concat", "external_python"} or primary_ablation != "none" or not checkpoint_info.get("supports_attention_kl", False):
            print("[table23_summary] Skipping attention KL: current policy source has no bottleneck attention export.", flush=True)
        else:
            attention_kl = {}
            for task_name, task_spec in heldout_specs.items():
                print(f"=== Attention KL for {args.label} on {task_name} ===", flush=True)
                attention_kl[task_name] = evaluate_task_attention_kl(
                    checkpoint_path=args.checkpoint,
                    data_root=args.data_root,
                    task_name=task_name,
                    task_root=task_spec["root"],
                    episodes=task_spec.get("episodes"),
                    device=args.device,
                    batch_size=args.batch_size,
                    load_rgb=args.attention_kl_load_rgb,
                    rgb_cache_dir=args.rgb_cache_dir,
                    max_pad_length=args.max_pad_length,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            payload["attention_kl"] = {
                "per_task": attention_kl,
                "summary": {
                    metric: float(np.mean([task_result[metric] for task_result in attention_kl.values()]))
                    for metric in ["kl_L1_L0", "kl_L2_L0", "kl_L2_L1"]
                },
            }

    if args.run_vlm_understanding:
        vlm_output_path = str(Path(args.output_dir) / "vlm_understanding.json")
        payload["vlm_understanding"] = evaluate_understanding(
            checkpoint_path=args.checkpoint,
            model_name=args.label,
            benchmarks=args.vlm_benchmarks,
            output_path=vlm_output_path,
            use_published=False,
            backend=args.vlm_backend,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "table23_summary.json"
    tactile_ablation_path = output_dir / "tactile_ablation_summary.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for variant_name, variant_payload in variant_payloads.items():
        suffix = "" if variant_name == "none" and len(variant_payloads) == 1 else f"_{variant_name}"
        table2_path = output_dir / f"table2_summary{suffix}.md"
        table3_path = output_dir / f"table3_summary{suffix}.md"
        table2_path.write_text(
            build_table2_markdown(
                f"{args.label} [{variant_name}]",
                args.checkpoint,
                checkpoint_epoch,
                variant_payload["per_task"],
                variant_payload["summary"],
            ),
            encoding="utf-8",
        )
        table3_path.write_text(
            build_table3_markdown(
                f"{args.label} [{variant_name}]",
                args.checkpoint,
                checkpoint_epoch,
                variant_payload["per_task"],
                variant_payload["summary"],
            ),
            encoding="utf-8",
        )
    if len(variant_payloads) > 1:
        tactile_ablation_path.write_text(
            build_tactile_ablation_markdown(
                checkpoint_label=args.label,
                variant_payloads=variant_payloads,
                deltas_vs_none=payload.get("ablation_deltas_vs_none", {}),
            ),
            encoding="utf-8",
        )

    print(f"saved {json_path}")
    if len(variant_payloads) > 1:
        print(f"saved {tactile_ablation_path}")


if __name__ == "__main__":
    main()