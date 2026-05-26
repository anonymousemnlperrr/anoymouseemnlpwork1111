from __future__ import annotations

import argparse
import json
from pathlib import Path

from VTLA.eval.stage_a_repr_eval import evaluate_stage_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a directory of Stage A checkpoints on held-out representation metrics")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--instruction-level", default="L2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-names", nargs="+", default=None)
    parser.add_argument("--phys-temperature", type=float, default=1.25)
    parser.add_argument("--phys-aggregation", choices=["none", "episode"], default="episode")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("stage_a_epoch*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No stage_a_epoch*.pt found in {checkpoint_dir}")

    results = []
    for checkpoint_path in checkpoint_paths:
        metrics = evaluate_stage_a(
            checkpoint_path=str(checkpoint_path),
            config_path=args.config,
            data_root=args.data_root,
            device=args.device,
            batch_size=args.batch_size,
            instruction_level=args.instruction_level,
            task_names=args.task_names,
            phys_temperature=args.phys_temperature,
            phys_aggregation=args.phys_aggregation,
        )
        metrics["checkpoint_name"] = checkpoint_path.name
        results.append(metrics)
        print(
            f"{checkpoint_path.name}: align_score={metrics['align_score']:.4f} "
            f"phys_accuracy={metrics['phys_accuracy']:.4f} cosine_gap={metrics['cosine_gap']:.4f}"
        )

    ranked = sorted(
        results,
        key=lambda item: (
            item["align_score"],
            item["phys_accuracy"],
            item["cosine_gap"],
        ),
        reverse=True,
    )

    payload = {
        "ranking_key": ["align_score", "phys_accuracy", "cosine_gap"],
        "best_checkpoint": ranked[0]["checkpoint"],
        "results": ranked,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()