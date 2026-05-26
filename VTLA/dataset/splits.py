"""
VTLA/dataset/splits.py

Task-level train/test splits for experiments.
默认优先按 folder 划分; 若只有单个 folder, 则退化为 episode-level held-out。
"""

from __future__ import annotations

from typing import Any


def _entry(name: str, root: str | None = None, episodes: list[int] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "root": root or name,
    }
    if episodes is not None:
        entry["episodes"] = episodes
    return entry


MATERIAL_PROBE_TRAIN_EPISODES = list(range(16))
MATERIAL_PROBE_HELDOUT_EPISODES = list(range(16, 20))

MATERIAL_PROBE_COTTON_TRAIN = {
    "Grabbing-Cotton-train-a": {
        "root": "Grabbing-Cotton-01",
        "episodes": [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14],
    },
    "Grabbing-Cotton-train-b": {
        "root": "Grabbing-Cotton-02",
        "episodes": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14],
    },
}
MATERIAL_PROBE_COTTON_TEST = {
    "Grabbing-Cotton-heldout-a": {
        "root": "Grabbing-Cotton-01",
        "episodes": [4, 10],
    },
    "Grabbing-Cotton-heldout-b": {
        "root": "Grabbing-Cotton-02",
        "episodes": [0, 1, 12, 13],
    },
}

MATERIAL_PROBE_SAND_TRAIN_EPISODES = list(range(16))
MATERIAL_PROBE_SAND_HELDOUT_EPISODES = list(range(16, 20))

MATERIAL_PROBE_SOYBEANS_TRAIN_EPISODES = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]
MATERIAL_PROBE_SOYBEANS_HELDOUT_EPISODES = [1, 2, 16, 17]


SPLITS = {
    "inboxpicking": {
        "train": [_entry(f"inboxpicking-{i:02d}") for i in range(1, 7)],
        "test": [_entry("inboxpicking-07")],
    },
    "grasp-blueberry": {
        "train": [_entry(f"grasp-blueberry-{i:02d}") for i in range(1, 5)],
        "test": [_entry("grasp-blueberry-05")],
    },
    "grasp-egg": {
        "train": [_entry("grasp-egg")],
        "test": [_entry("grasp-egg-02")],
    },
    "grasp-sponge": {
        "train": [_entry("grasp-Sponge-hard"), _entry("grasp-Sponge-soft-01")],
        "test": [_entry("grasp-Sponge-soft-02")],
    },
    "grabbing-cotton": {
        "train": [
            _entry(name, root=spec["root"], episodes=spec["episodes"])
            for name, spec in MATERIAL_PROBE_COTTON_TRAIN.items()
        ],
        "test": [
            _entry(name, root=spec["root"], episodes=spec["episodes"])
            for name, spec in MATERIAL_PROBE_COTTON_TEST.items()
        ],
    },
    "grabbing-sand": {
        "train": [
            _entry(
                "Grabbing-sand-train",
                root="Grabbing-sand",
                episodes=MATERIAL_PROBE_SAND_TRAIN_EPISODES,
            )
        ],
        "test": [
            _entry(
                "Grabbing-sand-heldout",
                root="Grabbing-sand",
                episodes=MATERIAL_PROBE_SAND_HELDOUT_EPISODES,
            )
        ],
    },
    "grabbing-soybeans": {
        "train": [
            _entry(
                "Grabbing-soybeans-train",
                root="Grabbing-soybeans",
                episodes=MATERIAL_PROBE_SOYBEANS_TRAIN_EPISODES,
            )
        ],
        "test": [
            _entry(
                "Grabbing-soybeans-heldout",
                root="Grabbing-soybeans",
                episodes=MATERIAL_PROBE_SOYBEANS_HELDOUT_EPISODES,
            )
        ],
    },
}


def _resolve_entries(entries: list[dict[str, Any]], base_dir: str) -> dict[str, dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    for entry in entries:
        root_spec: dict[str, Any] = {
            "root": f"{base_dir}/{entry['root']}",
        }
        if "episodes" in entry:
            root_spec["episodes"] = list(entry["episodes"])
        roots[entry["name"]] = root_spec
    return roots


def get_train_roots(base_dir: str = "data") -> dict[str, dict[str, Any]]:
    """返回所有训练任务的 {task_name: split_spec} 字典。"""
    roots: dict[str, dict[str, Any]] = {}
    for split in SPLITS.values():
        roots.update(_resolve_entries(split["train"], base_dir))
    return roots


def get_test_roots(base_dir: str = "data") -> dict[str, dict[str, Any]]:
    """返回所有 held-out 任务的 {task_name: split_spec} 字典。"""
    roots: dict[str, dict[str, Any]] = {}
    for split in SPLITS.values():
        roots.update(_resolve_entries(split["test"], base_dir))
    return roots


def get_named_roots(
    task_names: list[str] | None,
    base_dir: str = "data",
) -> dict[str, dict[str, Any]]:
    """Resolve explicit task names against both train/test aliases.

    This is intended for ad-hoc analysis runs only. Default official evaluation
    should still use ``get_test_roots`` with no manual task override.
    """
    train_roots = get_train_roots(base_dir)
    test_roots = get_test_roots(base_dir)
    all_roots = {**train_roots, **test_roots}
    if not task_names:
        return all_roots

    selected: dict[str, dict[str, Any]] = {}
    for task_name in task_names:
        if task_name in all_roots:
            selected[task_name] = dict(all_roots[task_name])
        else:
            selected[task_name] = {"root": f"{base_dir}/{task_name}"}
    return selected


def get_selected_test_roots(
    task_names: list[str] | None,
    base_dir: str = "data",
) -> dict[str, dict[str, Any]]:
    """Resolve an optional held-out subset while preserving split aliases.

    If a requested name matches a held-out alias in ``SPLITS``, its configured
    root and episode subset are preserved. Otherwise it falls back to a direct
    folder under ``base_dir`` for ad-hoc single-task analysis.
    """
    all_roots = get_test_roots(base_dir)
    if not task_names:
        return all_roots

    selected: dict[str, dict[str, Any]] = {}
    for task_name in task_names:
        if task_name in all_roots:
            selected[task_name] = dict(all_roots[task_name])
        else:
            selected[task_name] = {"root": f"{base_dir}/{task_name}"}
    return selected
