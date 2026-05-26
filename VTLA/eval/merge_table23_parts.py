from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from VTLA.eval.table23_summary import (
    build_tactile_ablation_markdown,
    build_table2_markdown,
    build_table3_markdown,
    summarize_checkpoint,
    summarize_ablation_deltas,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial Table 2/3 summary outputs into one final report")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    merged = {
        "checkpoint_label": None,
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "checkpoint_source_type": None,
        "tactile_ablations": ["none"],
        "per_task": {},
        "variants": {},
    }
    attention_kl_per_task = {}
    vlm_understanding = None

    tactile_ablation_order: list[str] = []

    for input_path in args.inputs:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        merged["checkpoint_label"] = merged["checkpoint_label"] or payload.get("checkpoint_label")
        merged["checkpoint_path"] = merged["checkpoint_path"] or payload.get("checkpoint_path")
        merged["checkpoint_epoch"] = merged["checkpoint_epoch"] or payload.get("checkpoint_epoch")
        merged["checkpoint_source_type"] = merged["checkpoint_source_type"] or payload.get("checkpoint_source_type")

        for tactile_ablation in payload.get("tactile_ablations", ["none"]):
            if tactile_ablation not in tactile_ablation_order:
                tactile_ablation_order.append(tactile_ablation)

        payload_variants = payload.get("variants")
        if isinstance(payload_variants, dict) and payload_variants:
            for variant_name, variant_payload in payload_variants.items():
                merged_variant = merged["variants"].setdefault(variant_name, {"per_task": {}})
                merged_variant["per_task"].update(variant_payload.get("per_task", {}))
        else:
            primary_ablation = payload.get("tactile_ablations", ["none"])[0]
            merged_variant = merged["variants"].setdefault(primary_ablation, {"per_task": {}})
            merged_variant["per_task"].update(payload.get("per_task", {}))

        attention_payload = payload.get("attention_kl", {})
        attention_kl_per_task.update(attention_payload.get("per_task", {}))
        if vlm_understanding is None and payload.get("vlm_understanding") is not None:
            vlm_understanding = payload["vlm_understanding"]

    if not tactile_ablation_order:
        tactile_ablation_order = sorted(merged["variants"].keys()) or ["none"]
    merged["tactile_ablations"] = tactile_ablation_order

    for variant_name, variant_payload in merged["variants"].items():
        variant_payload["summary"] = summarize_checkpoint(variant_payload["per_task"])

    primary_ablation = "none" if "none" in merged["variants"] else tactile_ablation_order[0]
    primary_payload = merged["variants"][primary_ablation]
    merged["per_task"] = primary_payload["per_task"]
    merged["summary"] = primary_payload["summary"]

    if "none" in merged["variants"]:
        merged["ablation_deltas_vs_none"] = {
            variant_name: summarize_ablation_deltas(merged["variants"]["none"]["summary"], variant_payload["summary"])
            for variant_name, variant_payload in merged["variants"].items()
            if variant_name != "none"
        }

    if attention_kl_per_task:
        merged["attention_kl"] = {
            "per_task": attention_kl_per_task,
            "summary": {
                metric: float(np.mean([task_result[metric] for task_result in attention_kl_per_task.values()]))
                for metric in ["kl_L1_L0", "kl_L2_L0", "kl_L2_L1"]
            },
        }

    if vlm_understanding is not None:
        merged["vlm_understanding"] = vlm_understanding

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "table23_summary.json"
    table2_path = output_dir / "table2_summary.md"
    table3_path = output_dir / "table3_summary.md"
    tactile_ablation_path = output_dir / "tactile_ablation_summary.md"

    json_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    table2_path.write_text(
        build_table2_markdown(
            merged["checkpoint_label"],
            merged["checkpoint_path"],
            merged["checkpoint_epoch"],
            merged["per_task"],
            merged["summary"],
        ),
        encoding="utf-8",
    )
    table3_path.write_text(
        build_table3_markdown(
            merged["checkpoint_label"],
            merged["checkpoint_path"],
            merged["checkpoint_epoch"],
            merged["per_task"],
            merged["summary"],
        ),
        encoding="utf-8",
    )

    if len(merged["variants"]) > 1:
        for variant_name, variant_payload in merged["variants"].items():
            variant_table2_path = output_dir / f"table2_summary_{variant_name}.md"
            variant_table3_path = output_dir / f"table3_summary_{variant_name}.md"
            variant_table2_path.write_text(
                build_table2_markdown(
                    f"{merged['checkpoint_label']} [{variant_name}]",
                    merged["checkpoint_path"],
                    merged["checkpoint_epoch"],
                    variant_payload["per_task"],
                    variant_payload["summary"],
                ),
                encoding="utf-8",
            )
            variant_table3_path.write_text(
                build_table3_markdown(
                    f"{merged['checkpoint_label']} [{variant_name}]",
                    merged["checkpoint_path"],
                    merged["checkpoint_epoch"],
                    variant_payload["per_task"],
                    variant_payload["summary"],
                ),
                encoding="utf-8",
            )

        tactile_ablation_path.write_text(
            build_tactile_ablation_markdown(
                checkpoint_label=merged["checkpoint_label"],
                variant_payloads=merged["variants"],
                deltas_vs_none=merged.get("ablation_deltas_vs_none", {}),
            ),
            encoding="utf-8",
        )

    print(f"saved {json_path}")
    print(f"saved {table2_path}")
    print(f"saved {table3_path}")
    if len(merged["variants"]) > 1:
        print(f"saved {tactile_ablation_path}")


if __name__ == "__main__":
    main()