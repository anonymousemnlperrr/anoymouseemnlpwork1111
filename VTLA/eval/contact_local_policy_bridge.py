"""External eval bridge for B2 Contact-Local Semantic Actuation checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from VTLA.models.contact_local_routing import (
    build_contact_local_context_from_entries,
    build_contact_metadata_index,
    compute_online_contact_score,
    normalize_contact_local_cfg,
)
from VTLA.models.contact_local_policy import (
    B2ContactLocalSemanticActuationPolicy,
)


class B2CLSAEvalBridge:
    def __init__(self, policy: B2ContactLocalSemanticActuationPolicy, contact_cfg: dict[str, float] | None = None) -> None:
        self.policy = policy
        self.contact_cfg = normalize_contact_local_cfg(contact_cfg)
        self.processor = policy.processor
        self.tokenizer = policy.processor.tokenizer
        self._contact_metadata_cache: dict[tuple[str, str], dict[tuple[str, int], dict[str, float]]] = {}
        self._frame_bounds_cache: dict[tuple[str, str], dict[tuple[str, int], tuple[int, int]]] = {}

    def to(self, device: torch.device | str) -> "B2CLSAEvalBridge":
        self.policy.to(device)
        return self

    def eval(self) -> "B2CLSAEvalBridge":
        self.policy.eval()
        return self

    def parameters(self):
        return self.policy.parameters()

    def _policy_device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def _as_list(self, values):
        if values is None:
            return None
        if isinstance(values, (str, Path)):
            return [str(values)]
        if torch.is_tensor(values):
            return values.detach().cpu().tolist()
        if isinstance(values, (int, float, bool)):
            return [values]
        try:
            return list(values)
        except TypeError:
            return [values]

    def _get_task_contact_context(
        self,
        task_name: str,
        task_root: str,
        frame_bounds: dict[int, tuple[int, int]] | None,
    ) -> tuple[dict[tuple[str, int], dict[str, float]], dict[tuple[str, int], tuple[int, int]]]:
        cache_key = (str(task_name), str(task_root))
        if cache_key not in self._contact_metadata_cache:
            self._contact_metadata_cache[cache_key] = build_contact_metadata_index({task_name: {"root": task_root}})
        bounds_cache = self._frame_bounds_cache.setdefault(cache_key, {})
        if frame_bounds:
            for episode_index, episode_bounds in frame_bounds.items():
                frame_min, frame_max = episode_bounds
                bounds_cache[(str(task_name), int(episode_index))] = (int(frame_min), int(frame_max))
        return self._contact_metadata_cache[cache_key], bounds_cache

    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tactile_grid: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        task_names = self._as_list(kwargs.get("task_names"))
        episode_indices = self._as_list(kwargs.get("episode_indices"))
        frame_indices = self._as_list(kwargs.get("frame_indices"))
        task_root = kwargs.get("task_root")
        frame_bounds = kwargs.get("frame_bounds")

        device = self._policy_device()
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        tactile_grid = tactile_grid.to(device=device)
        residual_mask = None
        anchor_mask = None
        contact_features = None

        if task_names and episode_indices and frame_indices and task_root and frame_bounds is not None:
            unique_task_names = {str(name) for name in task_names}
            if len(unique_task_names) == 1:
                task_name = next(iter(unique_task_names))
                contact_metadata, task_frame_bounds = self._get_task_contact_context(
                    task_name=task_name,
                    task_root=str(task_root),
                    frame_bounds=frame_bounds,
                )
                contact_ctx = build_contact_local_context_from_entries(
                    task_names=task_names,
                    episode_indices=episode_indices,
                    frame_indices=frame_indices,
                    tactile_grid=tactile_grid,
                    contact_metadata=contact_metadata,
                    frame_bounds=task_frame_bounds,
                    cfg=self.contact_cfg,
                )
                residual_mask = contact_ctx["residual_mask"]
                anchor_mask = contact_ctx["anchor_mask"]
                contact_features = contact_ctx["contact_features"]

        if residual_mask is None:
            online_score = compute_online_contact_score(
                tactile_grid,
                peak_threshold=float(self.contact_cfg["online_contact_peak_threshold"]),
            ).to(device=device, dtype=torch.float32)
            residual_mask = online_score
            anchor_mask = 1.0 - online_score
            contact_features = torch.stack([online_score, online_score, torch.zeros_like(online_score)], dim=-1)

        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            image_grid_thw = image_grid_thw.to(device=device)

        return self.policy.predict_action(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            residual_mask=residual_mask,
            anchor_mask=anchor_mask,
            contact_features=contact_features,
        )


def build_policy(checkpoint_path: str) -> tuple[B2CLSAEvalBridge, dict[str, Any]]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = B2ContactLocalSemanticActuationPolicy(
        vlm_model_id=str(checkpoint.get("vlm_model_id", "Qwen/Qwen2-VL-2B-Instruct")),
        action_dim=int(checkpoint.get("action_dim", 6)),
        chunk_size=int(checkpoint.get("chunk_size", 1)),
        n_queries=int(checkpoint.get("n_queries", 16)),
        d_bottleneck=int(checkpoint.get("d_bottleneck", 512)),
        n_bottleneck_layers=int(checkpoint.get("n_bottleneck_layers", 2)),
        lora_rank=int(checkpoint.get("lora_rank", 16)),
        lora_alpha=float(checkpoint.get("lora_alpha", 32.0)),
        freeze_vlm=True,
        freeze_encoders=bool(checkpoint.get("freeze_encoders", True)),
        disable_tactile=bool(checkpoint.get("disable_tactile", False)),
        disable_rgb=bool(checkpoint.get("disable_rgb", False)),
        bottleneck_variant=str(checkpoint.get("bottleneck_variant", "lang_guided")),
        residual_action_scale=float(checkpoint.get("residual_action_scale", 0.10)),
        min_alpha=float(checkpoint.get("min_alpha", 0.0)),
        max_alpha=float(checkpoint.get("max_alpha", 1.0)),
        tactile_confidence_bias_init=float(checkpoint.get("tactile_confidence_bias_init", -2.0)),
        monotonic_margin=float(checkpoint.get("monotonic_margin", 0.02)),
    )
    policy.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    policy.eval()
    bridge = B2CLSAEvalBridge(policy=policy, contact_cfg=normalize_contact_local_cfg(checkpoint))
    metadata = {
        "load_rgb": bool(not checkpoint.get("disable_rgb", False)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "supports_attention_kl": False,
        "external_checkpoint_path": checkpoint_path,
        "external_policy_family": "b2_contact_local_semantic_actuation",
    }
    return bridge, metadata
