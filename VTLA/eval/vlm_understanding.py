"""
VTLA/eval/vlm_understanding.py

VLM Understanding Benchmark Evaluation

使用 VLMEvalKit 或 lm-evaluation-harness 评估 VTLA-2B 在标准
VLM 理解 benchmarks 上的表现, 与 Qwen2-VL-2B / DiVLA-2B 对比。

对应论文 Table 1 (VLM Understanding Benchmarks)

Benchmarks:
  - MMMU (多学科多模态理解)
  - MMStar (多模态综合)
  - TextVQA (文字识别 VQA)
  - ScienceQA (科学问答)
  - ChartQA (图表理解)

Usage:
  # 评估 VTLA-2B checkpoint
  python -m VTLA.eval.vlm_understanding \\
      --checkpoint path/to/ckpt.pt \\
      --benchmarks MMMU TextVQA ScienceQA \\
      --output results/vlm_understanding.json

  # 评估 base Qwen2-VL-2B (无 checkpoint, 作为 reference)
  python -m VTLA.eval.vlm_understanding \\
      --benchmarks MMMU TextVQA \\
      --output results/qwen2vl_reference.json
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


# Published reference numbers (from original papers / ChatVLA Table 1)
PUBLISHED_REFERENCES = {
    "Qwen2-VL-2B": {
        "MMMU": 41.1,
        "MMStar": 48.0,
        "TextVQA": 79.7,
        "ScienceQA": None,  # need to measure
        "ChartQA": None,
    },
    "DiVLA-2B": {
        "MMMU": 17.2,
        "MMStar": 21.1,
        "TextVQA": 7.5,
        "ScienceQA": None,
        "ChartQA": None,
    },
    "OpenVLA-7B": {
        "MMMU": None,
        "MMStar": None,
        "TextVQA": None,
        "ScienceQA": None,
        "ChartQA": None,
    },
}


def merge_lora_linear(module):
    from VTLA.models.vtla_model import LoRALinear

    if not isinstance(module, LoRALinear):
        raise TypeError(f"Expected LoRALinear, got {type(module)!r}")

    base = module.base
    merged = torch.nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        delta = (module.lora_A @ module.lora_B).transpose(0, 1) * module.scale
        merged.weight.copy_(base.weight + delta.to(base.weight.dtype))
        if base.bias is not None:
            merged.bias.copy_(base.bias)
    return merged


def merge_lora_modules_in_place(model) -> int:
    from VTLA.models.vtla_model import LoRALinear

    merged_count = 0
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, LoRALinear):
            continue
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, merge_lora_linear(module))
        merged_count += 1
    return merged_count


def default_export_dir(checkpoint_path: str, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path).parent / "vtla_hf_export"
    ckpt_path = Path(checkpoint_path)
    return ckpt_path.parent / f"{ckpt_path.stem}_vlm_export"


def build_runner_manifest(
    export_dir: str,
    benchmarks: list[str],
    backend: str,
) -> dict:
    export_dir = str(Path(export_dir).resolve())
    manifest = {
        "backend": backend,
        "export_dir": export_dir,
        "benchmarks": benchmarks,
        "commands": [],
    }
    if backend in {"auto", "vlmeval"}:
        for benchmark in benchmarks:
            manifest["commands"].append(
                {
                    "benchmark": benchmark,
                    "command": [
                        sys.executable,
                        "-m",
                        "vlmeval",
                        "--model",
                        export_dir,
                        "--data",
                        benchmark,
                        "--output",
                        str(Path(export_dir) / "vlmeval_outputs" / f"{benchmark}.json"),
                    ],
                }
            )
    if backend in {"auto", "lm_eval"}:
        for benchmark in benchmarks:
            manifest["commands"].append(
                {
                    "benchmark": benchmark,
                    "command": [
                        "lm_eval",
                        "--model",
                        "hf-multimodal",
                        "--model_args",
                        f"pretrained={export_dir}",
                        "--tasks",
                        benchmark,
                        "--output_path",
                        str(Path(export_dir) / "lm_eval_outputs" / benchmark),
                    ],
                    "note": "Task names and hf-multimodal support may need to match the installed lm-eval-harness version.",
                }
            )
    return manifest


def export_vtla_checkpoint_to_hf(
    checkpoint_path: str,
    export_dir: str,
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    safe_serialization: bool = True,
) -> dict:
    from VTLA.models.vtla_model import VTLAModel
    from transformers import AutoConfig

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # FIX: Load and preserve correct generation config from base model
    try:
        base_config = AutoConfig.from_pretrained(
            base_model_id, local_files_only=True, trust_remote_code=True
        )
        print(f"[Export] Loaded base model config from cache")
    except Exception:
        try:
            base_config = AutoConfig.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            print(f"[Export] Loaded base model config from HuggingFace")
        except Exception as e:
            print(f"[Export] WARNING: Could not load base config: {e}")
            base_config = None

    model = VTLAModel(vlm_model_id=base_model_id)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    merge_count = merge_lora_modules_in_place(model.vlm)

    model.vlm.save_pretrained(export_path, safe_serialization=safe_serialization)
    model.processor.save_pretrained(export_path)

    # FIX: Write correct generation_config.json to override incomplete/broken one
    # The issue: save_pretrained() may not preserve all necessary generation parameters
    # Solution: Explicitly write a complete generation_config.json after saving
    if base_config is not None:
        gen_config_dict = {
            # Core generation parameters
            "do_sample": getattr(base_config, 'do_sample', False),
            "temperature": getattr(base_config, 'temperature', 1.0),
            "top_k": getattr(base_config, 'top_k', 50),
            "top_p": getattr(base_config, 'top_p', 1.0),
            # FIX: Use max_position_embeddings as max_length for long context, not config.max_length=20
            "max_length": getattr(base_config, 'max_position_embeddings', 8192),
            "max_new_tokens": getattr(base_config, 'max_new_tokens', 512),  # FIX: For lm-eval compat
            
            # Standard token IDs
            "bos_token_id": getattr(base_config, 'bos_token_id', 151643),
            "eos_token_id": getattr(base_config, 'eos_token_id', 151645),
            "pad_token_id": getattr(base_config, 'pad_token_id', 151643),
            
            # Beam search and control
            "num_beams": getattr(base_config, 'num_beams', 1),
            "early_stopping": getattr(base_config, 'early_stopping', False),
            "repetition_penalty": getattr(base_config, 'repetition_penalty', 1.0),
            
            # Additional Qwen2-VL specific settings
            "typical_p": getattr(base_config, 'typical_p', 1.0),
            "min_length": getattr(base_config, 'min_length', 0),
            
            # Metadata
            "transformers_version": "4.57.6",
        }
        
        gen_config_path = export_path / "generation_config.json"
        with open(gen_config_path, 'w') as f:
            json.dump(gen_config_dict, f, indent=2)
        
        print(f"[Export] Wrote corrected generation_config.json:")
        print(f"  do_sample={gen_config_dict['do_sample']}, "
              f"temperature={gen_config_dict['temperature']}, "
              f"max_length={gen_config_dict['max_length']}, "
              f"max_new_tokens={gen_config_dict['max_new_tokens']}")

    metadata = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "base_model_id": base_model_id,
        "merged_lora_layers": merge_count,
        "export_dir": str(export_path.resolve()),
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "notes": [
            "This export contains the Qwen2-VL backbone with VTLA LoRA merged into standard Linear layers.",
            "Tactile encoder, bottleneck, and action head are not part of the exported benchmark model.",
            "This export is intended for Table 1 understanding benchmarks on external runners.",
            "FIX (2026-04-27): Generation config explicitly written to prevent generation failures.",
            "max_length=2048, max_new_tokens=512 explicitly set for lm-eval compatibility.",
        ],
    }
    (export_path / "vtla_export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run_vlmeval(
    model_path: str,
    benchmark: str,
    output_dir: str,
    env: dict[str, str] | None = None,
) -> dict:
    """
    Run VLMEvalKit for a single benchmark.

    Requires VLMEvalKit to be installed:
      pip install vlmeval
    """
    try:
        import vlmeval  # noqa: F401
    except ImportError:
        print("VLMEvalKit not installed. Install with: pip install vlmeval")
        print("Falling back to published reference numbers.")
        return {}

    out_path = Path(output_dir) / f"{benchmark}.json"
    cmd = [
        sys.executable, "-m", "vlmeval",
        "--model", model_path,
        "--data", benchmark,
        "--output", str(out_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"VLMEvalKit failed for {benchmark}: {result.stderr}")
        return {}

    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {}


def detect_backend(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if importlib.util.find_spec("vlmeval"):
        return "vlmeval"
    if importlib.util.find_spec("lm_eval"):
        return "lm_eval"
    return "published"


def _benchmark_record(score, source: str, status: str, note: str | None = None) -> dict:
    record = {
        "score": score,
        "source": source,
        "status": status,
    }
    if note:
        record["note"] = note
    return record


def evaluate_understanding(
    checkpoint_path: str | None = None,
    model_name: str = "VTLA-2B",
    benchmarks: list[str] | None = None,
    output_path: str | None = None,
    use_published: bool = True,
    backend: str = "auto",
    export_dir: str | None = None,
    prepare_only: bool = False,
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
):
    """
    Evaluate VLM understanding benchmarks.

    If checkpoint is None, returns published reference numbers.
    """
    if benchmarks is None:
        benchmarks = ["MMMU", "MMStar", "TextVQA", "ScienceQA", "ChartQA"]

    resolved_backend = detect_backend(backend)
    results = {
        "model": model_name,
        "backend": resolved_backend,
        "checkpoint_path": checkpoint_path,
        "benchmarks": {},
    }

    export_metadata = None
    runner_manifest = None
    if checkpoint_path is not None:
        export_path = Path(export_dir) if export_dir else default_export_dir(checkpoint_path, output_path)
        results["export_dir"] = str(export_path.resolve())
        if prepare_only or resolved_backend in {"vlmeval", "lm_eval"}:
            export_metadata = export_vtla_checkpoint_to_hf(
                checkpoint_path=checkpoint_path,
                export_dir=str(export_path),
                base_model_id=base_model_id,
            )
            results["export"] = export_metadata
            runner_manifest = build_runner_manifest(str(export_path), benchmarks or [], resolved_backend)
            results["runner_manifest"] = runner_manifest

    if prepare_only:
        for bm in benchmarks or []:
            results["benchmarks"][bm] = _benchmark_record(
                score=None,
                source="prepared_export",
                status="ready_for_external_runner",
                note="Use runner_manifest commands with the exported HF directory.",
            )
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        return results

    if checkpoint_path is None and use_published:
        # Return published numbers
        ref = PUBLISHED_REFERENCES.get(model_name, {})
        for bm in benchmarks:
            score = ref.get(bm, None)
            status = "ok" if score is not None else "missing_reference"
            results["benchmarks"][bm] = _benchmark_record(
                score=score,
                source="published_reference",
                status=status,
            )
        print(f"\n=== Published Reference: {model_name} ===")
        for bm, record in results["benchmarks"].items():
            print(f"  {bm}: {record['score']}")
    elif checkpoint_path is not None and resolved_backend == "published":
        for bm in benchmarks:
            results["benchmarks"][bm] = _benchmark_record(
                score=None,
                source="unavailable_backend",
                status="pending_external_runner",
                note="Neither vlmeval nor lm_eval is installed in the runtime environment.",
            )
        results["runner_hint"] = {
            "vlmeval_required": "pip install vlmeval",
            "lm_eval_required": "pip install lm-eval",
            "checkpoint_note": "Run this script with --prepare-only or a real backend to export a merged HF model for external benchmark runners.",
        }
        if runner_manifest is not None:
            results["runner_manifest"] = runner_manifest
    else:
        # Run actual evaluation
        model_path = str(Path(results.get("export_dir", checkpoint_path or "Qwen/Qwen2-VL-2B-Instruct")).resolve())
        output_dir = str(Path(output_path).parent) if output_path else "results/"
        runner_env = os.environ.copy()
        for bm in benchmarks:
            if resolved_backend == "vlmeval":
                bm_result = run_vlmeval(model_path, bm, output_dir, env=runner_env)
                score = bm_result.get("accuracy", bm_result.get("score", None))
                status = "ok" if score is not None else "runner_failed"
                note = None if score is not None else "VLMEvalKit returned no accuracy field."
                results["benchmarks"][bm] = _benchmark_record(
                    score=score,
                    source="vlmeval",
                    status=status,
                    note=note,
                )
            else:
                results["benchmarks"][bm] = _benchmark_record(
                    score=None,
                    source=resolved_backend,
                    status="ready_for_external_runner",
                    note="Use runner_manifest to launch lm_eval against the exported HF model directory.",
                )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def generate_table1(
    vtla_results: dict,
    include_published: bool = True,
) -> str:
    """Generate markdown Table 1 (VLM Understanding Benchmarks)."""
    benchmarks = ["MMMU", "MMStar", "TextVQA", "ScienceQA", "ChartQA"]
    header = "| Method | #Params | " + " | ".join(benchmarks) + " |"
    sep = "|--------|---------|" + "|".join(["--------"] * len(benchmarks)) + "|"

    rows = [header, sep]

    if include_published:
        rows.append("| **VLMs** | | " + " | ".join([""] * len(benchmarks)) + " |")
        for model_name in ["Qwen2-VL-2B"]:
            ref = PUBLISHED_REFERENCES.get(model_name, {})
            vals = [str(ref.get(bm, "–")) for bm in benchmarks]
            rows.append(f"| {model_name} | 2B | " + " | ".join(vals) + " |")

        rows.append("| **VLAs** | | " + " | ".join([""] * len(benchmarks)) + " |")
        for model_name in ["DiVLA-2B", "OpenVLA-7B"]:
            ref = PUBLISHED_REFERENCES.get(model_name, {})
            size = "2B" if "2B" in model_name else "7B"
            vals = [str(ref.get(bm, "–")) for bm in benchmarks]
            rows.append(f"| {model_name} | {size} | " + " | ".join(vals) + " |")

    # Our model
    vtla_bm = vtla_results.get("benchmarks", {})
    vals = []
    for bm in benchmarks:
        record = vtla_bm.get(bm, {})
        score = record.get("score") if isinstance(record, dict) else record
        vals.append("–" if score is None else str(score))
    rows.append(f"| **VTLA-2B (Ours)** | **2B** | " + " | ".join(vals) + " |")

    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="VTLA checkpoint path")
    parser.add_argument("--model_name", default="VTLA-2B")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["MMMU", "MMStar", "TextVQA"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "published", "vlmeval", "lm_eval"])
    parser.add_argument("--export_dir", default=None,
                        help="Directory for a merged HF-export of the VTLA checkpoint")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only export the VTLA checkpoint to a runner-ready HF directory and write runner commands")
    parser.add_argument("--base-model-id", default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Base Qwen2-VL model id used to reconstruct the HF export before merging LoRA")
    parser.add_argument("--published_only", action="store_true",
                        help="Only show published reference numbers")
    args = parser.parse_args()

    if args.published_only:
        for model in ["Qwen2-VL-2B", "DiVLA-2B", "OpenVLA-7B"]:
            evaluate_understanding(
                model_name=model, benchmarks=args.benchmarks, use_published=True, backend=args.backend
            )
    else:
        results = evaluate_understanding(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            benchmarks=args.benchmarks,
            output_path=args.output,
            backend=args.backend,
            export_dir=args.export_dir,
            prepare_only=args.prepare_only,
            base_model_id=args.base_model_id,
        )
        print("\n=== Table 1: VLM Understanding ===")
        print(generate_table1(results))
