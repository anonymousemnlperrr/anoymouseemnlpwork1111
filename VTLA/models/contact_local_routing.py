from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_CONTACT_LOCAL_CFG: dict[str, float] = {
    "online_contact_peak_threshold": 20.0,
    "metadata_contact_ratio_min": 0.02,
    "metadata_contact_ratio_max": 0.95,
    "metadata_active_ratio_min": 0.10,
    "metadata_start_margin_frames": 2.0,
    "metadata_contact_weight": 0.85,
    "max_precontact_online": 0.05,
    "pre_contact_prior": 0.0,
    "in_contact_prior": 1.0,
    "post_contact_prior": 0.25,
    "contact_tail_frames": 2.0,
}


PHASE_NAMES = ("approach", "contact", "lift", "retreat")


def normalize_contact_local_cfg(source: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = dict(DEFAULT_CONTACT_LOCAL_CFG)
    if source is None:
        return cfg
    for key, default in DEFAULT_CONTACT_LOCAL_CFG.items():
        if key in source:
            cfg[key] = float(source.get(key, default))
    return cfg


def _resolve_root_spec(root_spec: Any) -> tuple[Path, set[int] | None]:
    if isinstance(root_spec, dict):
        root = Path(root_spec["root"])
        episodes = root_spec.get("episodes")
        if episodes is None:
            return root, None
        return root, {int(value) for value in episodes}
    return Path(str(root_spec)), None


def build_contact_metadata_index(data_roots: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    contact_metadata: dict[tuple[str, int], dict[str, float]] = {}
    for task_name, root_spec in data_roots.items():
        task_root, allowed_episodes = _resolve_root_spec(root_spec)
        metadata_path = task_root / "episode_metadata.json"
        if not metadata_path.exists():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes", {})
        for episode_key, episode_payload in episodes.items():
            episode_index = int(episode_key)
            if allowed_episodes is not None and episode_index not in allowed_episodes:
                continue
            features = episode_payload.get("features", {})
            contact_metadata[(str(task_name), episode_index)] = {
                "first_contact_frame": float(features.get("first_contact_frame", episode_payload.get("n_frames", 0))),
                "last_contact_frame": float(features.get("last_contact_frame", episode_payload.get("n_frames", 0))),
                "contact_ratio": float(features.get("contact_ratio", 0.0)),
                "contact_mean_active_ratio": float(features.get("contact_mean_active_ratio", 0.0)),
                "contact_max_active_ratio": float(features.get("contact_max_active_ratio", 0.0)),
                "contact_mean_force": float(features.get("contact_mean_force", 0.0)),
                "contact_peak_force": float(features.get("contact_peak_force", 0.0)),
                "force_gradient": float(features.get("force_gradient", 0.0)),
                "total_frames": float(features.get("total_frames", episode_payload.get("n_frames", 0))),
            }
    return contact_metadata


def build_frame_bounds(dataset) -> dict[tuple[str, int], tuple[int, int]]:
    bounds: dict[tuple[str, int], tuple[int, int]] = {}
    children = getattr(dataset, "datasets", None)
    if children is not None:
        for child in children:
            bounds.update(build_frame_bounds(child))
        return bounds

    task_name = getattr(dataset, "task_name", None)
    frame_bounds = getattr(dataset, "_frame_bounds", None)
    if task_name is None or frame_bounds is None:
        return bounds

    for episode_index, episode_bounds in frame_bounds.items():
        frame_min, frame_max = episode_bounds
        bounds[(str(task_name), int(episode_index))] = (int(frame_min), int(frame_max))
    return bounds


def compute_online_contact_score(
    tactile_grid: torch.Tensor,
    peak_threshold: float = 20.0,
) -> torch.Tensor:
    tactile_abs = tactile_grid.detach().abs().to(torch.float32)
    frame_peaks = tactile_abs.amax(dim=(-1, -2, -3))
    contact_frames = (frame_peaks > peak_threshold).to(torch.float32)
    return (0.7 * contact_frames[:, -1] + 0.3 * contact_frames.mean(dim=1)).clamp(0.0, 1.0)


def infer_phase_name(frame_index: int, frame_bounds: tuple[int, int]) -> str:
    frame_min, frame_max = frame_bounds
    denom = max(frame_max - frame_min, 1)
    progress = (frame_index - frame_min) / denom
    if progress < 0.25:
        return "approach"
    if progress < 0.50:
        return "contact"
    if progress < 0.75:
        return "lift"
    return "retreat"


def infer_metadata_contact_prior(
    metadata_stats: dict[str, float] | None,
    frame_index: int,
    frame_bounds: tuple[int, int],
    cfg: dict[str, Any],
) -> tuple[float | None, bool]:
    if metadata_stats is None:
        return None, False

    frame_min, frame_max = frame_bounds
    first_contact = int(metadata_stats.get("first_contact_frame", frame_max))
    last_contact = int(metadata_stats.get("last_contact_frame", frame_max))
    contact_ratio = float(metadata_stats.get("contact_ratio", 0.0))
    active_ratio = float(metadata_stats.get("contact_mean_active_ratio", 0.0))

    reliable = (
        contact_ratio >= float(cfg.get("metadata_contact_ratio_min", DEFAULT_CONTACT_LOCAL_CFG["metadata_contact_ratio_min"]))
        and contact_ratio <= float(cfg.get("metadata_contact_ratio_max", DEFAULT_CONTACT_LOCAL_CFG["metadata_contact_ratio_max"]))
        and active_ratio >= float(cfg.get("metadata_active_ratio_min", DEFAULT_CONTACT_LOCAL_CFG["metadata_active_ratio_min"]))
        and first_contact > frame_min + int(cfg.get("metadata_start_margin_frames", DEFAULT_CONTACT_LOCAL_CFG["metadata_start_margin_frames"]))
        and last_contact >= first_contact
    )
    if not reliable:
        return None, False

    if frame_index < first_contact:
        return float(cfg.get("pre_contact_prior", DEFAULT_CONTACT_LOCAL_CFG["pre_contact_prior"])), True
    if frame_index <= last_contact:
        return float(cfg.get("in_contact_prior", DEFAULT_CONTACT_LOCAL_CFG["in_contact_prior"])), True

    post_contact_prior = float(cfg.get("post_contact_prior", DEFAULT_CONTACT_LOCAL_CFG["post_contact_prior"]))
    tail_frames = max(1, int(cfg.get("contact_tail_frames", DEFAULT_CONTACT_LOCAL_CFG["contact_tail_frames"])))
    post_offset = max(0, int(frame_index) - last_contact)
    decay = max(0.0, 1.0 - (float(post_offset) / float(tail_frames)))
    return post_contact_prior * decay, True


def blend_contact_prior(metadata_prior: float, online_score: float, cfg: dict[str, Any]) -> float:
    metadata_weight = float(cfg.get("metadata_contact_weight", DEFAULT_CONTACT_LOCAL_CFG["metadata_contact_weight"]))
    if metadata_prior <= 0.0:
        return min(float(online_score), float(cfg.get("max_precontact_online", DEFAULT_CONTACT_LOCAL_CFG["max_precontact_online"])))
    if metadata_prior >= 1.0:
        return max(float(online_score), metadata_weight)
    blended = metadata_weight * metadata_prior + (1.0 - metadata_weight) * float(online_score)
    return float(max(0.0, min(1.0, blended)))


def build_contact_local_context_from_entries(
    task_names,
    episode_indices,
    frame_indices,
    tactile_grid: torch.Tensor,
    contact_metadata: dict[tuple[str, int], dict[str, float]],
    frame_bounds: dict[tuple[str, int], tuple[int, int]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_contact_local_cfg(cfg)
    online_scores = compute_online_contact_score(
        tactile_grid,
        peak_threshold=float(cfg.get("online_contact_peak_threshold", DEFAULT_CONTACT_LOCAL_CFG["online_contact_peak_threshold"])),
    )

    residual_masks: list[float] = []
    anchor_masks: list[float] = []
    blended_priors: list[float] = []
    metadata_masks: list[float] = []
    phase_names: list[str] = []

    for task_name, episode_index, frame_index, online_score in zip(
        task_names,
        episode_indices,
        frame_indices,
        online_scores.tolist(),
    ):
        task_name = str(task_name)
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        bounds = frame_bounds.get((task_name, episode_index), (frame_index, frame_index))
        phase_names.append(infer_phase_name(frame_index, bounds))

        metadata_stats = contact_metadata.get((task_name, episode_index))
        metadata_prior, metadata_reliable = infer_metadata_contact_prior(metadata_stats, frame_index, bounds, cfg)
        if metadata_reliable and metadata_prior is not None:
            blended = blend_contact_prior(metadata_prior, online_score, cfg)
            metadata_masks.append(1.0)
        else:
            blended = float(online_score)
            metadata_masks.append(0.0)
        residual_masks.append(blended)
        anchor_masks.append(1.0 - blended)
        blended_priors.append(blended)

    device = tactile_grid.device
    residual_mask = torch.tensor(residual_masks, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    anchor_mask = torch.tensor(anchor_masks, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    metadata_mask = torch.tensor(metadata_masks, device=device, dtype=torch.float32)
    blended_prior = torch.tensor(blended_priors, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    contact_features = torch.stack([residual_mask, blended_prior, metadata_mask], dim=-1)
    return {
        "residual_mask": residual_mask,
        "anchor_mask": anchor_mask,
        "contact_prior": blended_prior,
        "online_score": online_scores.to(device=device, dtype=torch.float32),
        "metadata_mask": metadata_mask,
        "contact_features": contact_features,
        "phase_names": phase_names,
    }


def build_contact_local_context(
    batch: dict[str, Any],
    contact_metadata: dict[tuple[str, int], dict[str, float]],
    frame_bounds: dict[tuple[str, int], tuple[int, int]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_contact_local_context_from_entries(
        task_names=batch.get("task_name", []),
        episode_indices=batch.get("episode_index", []),
        frame_indices=batch.get("frame_index", []),
        tactile_grid=batch["tactile_grid"],
        contact_metadata=contact_metadata,
        frame_bounds=frame_bounds,
        cfg=cfg,
    )