"""
Post-hoc normalization of table23_summary.json results.

Divides raw-action MSE by the training-set mean variance so all models
are reported in a consistent normalized action space (z-score per-dimension).

Two normalization references:
  - VTLA training stats:   sigma2 = 808.08  (17 tasks, 126,778 frames)
  - pi05 training stats:    sigma2 = 1200.10  (from pi05 norm_stats.json)

For cross-model comparison, VTLA raw MSE is divided by VTLA sigma2,
and pi05's already-internally-normalized MSE is rescaled by the ratio
pi05_sigma2 / VTLA_sigma2 so both land in the same VTLA-normalized space.
"""

import json
import sys
from pathlib import Path

# VTLA training set action variance (avg of per-dimension variances)
# dim stds: [17.18, 44.29, 45.16, 15.30, 14.12, 10.91]
VTLA_SIGMA2 = 808.0825

# pi05 training set action variance (avg of per-dimension variances)
# dim stds: [24.29, 53.27, 53.22, 17.81, 22.33, 11.37]
PI05_SIGMA2 = 1200.10

# Rescale factor: pi05 internally-normalized MSE → VTLA-normalized space
PI05_TO_VTLA_SCALE = PI05_SIGMA2 / VTLA_SIGMA2  # ≈ 1.485

MSE_KEYS = ("MSE_L0", "MSE_L1", "MSE_L2", "overall_action_mse")
DELTA_KEYS = ("delta_L2_L0", "delta_L1_L0")


def _scale_subtree(obj: dict, scale: float, mse_keys: tuple, delta_keys: tuple) -> None:
    """Scale MSE / delta values in a nested dict (mutates in-place)."""
    for key in mse_keys:
        if key in obj:
            obj[key] = obj[key] / scale
    for key in delta_keys:
        if key in obj:
            obj[key] = obj[key] / scale


def normalize_results(data: dict, scale: float, source: str = "") -> dict:
    """Normalize MSE values in-place. Returns the modified dict."""
    # Per-task
    for task_data in data.get("per_task", {}).values():
        _scale_subtree(task_data, scale, MSE_KEYS, DELTA_KEYS)
        for phase_data in task_data.get("per_phase", {}).values():
            _scale_subtree(phase_data, scale, MSE_KEYS, DELTA_KEYS)

    # Per-family
    for family_data in data.get("per_family", {}).values():
        _scale_subtree(family_data, scale, MSE_KEYS, DELTA_KEYS)

    # Family-balanced
    if "family_balanced" in data:
        _scale_subtree(data["family_balanced"], scale, MSE_KEYS, DELTA_KEYS)

    # Per-phase macro
    for phase_data in data.get("per_phase_macro", {}).values():
        _scale_subtree(phase_data, scale, MSE_KEYS, DELTA_KEYS)

    data["_normalized"] = True
    data["_scale_factor"] = scale
    if source:
        data["_source"] = source
    return data


def normalize_pi05_to_vtla_space(data: dict) -> dict:
    """Rescale pi05 internally-normalized MSE to VTLA-normalized space."""
    return normalize_results(data, 1.0 / PI05_TO_VTLA_SCALE,
                             source=f"pi05→VTLA (×{PI05_TO_VTLA_SCALE:.3f})")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python normalize_mse.py <table23_summary.json> [...]")
        print("  python normalize_mse.py --pi05 <table23_summary.json> [...]")
        sys.exit(1)

    args = sys.argv[1:]
    pi05_mode = False
    if args[0] == "--pi05":
        pi05_mode = True
        args = args[1:]

    for path_str in args:
        path = Path(path_str)
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        if data.get("_normalized"):
            print(f"SKIP (already normalized): {path}")
            continue

        if pi05_mode:
            data = normalize_pi05_to_vtla_space(data)
        else:
            data = normalize_results(data, VTLA_SIGMA2, source="VTLA")

        out_path = path.parent / f"{path.stem}_normalized{path.suffix}"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"OK: {path} -> {out_path.name}")


if __name__ == "__main__":
    main()
