from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from VTLA.dataset.material_labels import MATERIAL_CLASSES, decode_material_label


@dataclass(slots=True)
class MaterialProbePredictionResult:
    episode_id: str
    pred_material: str
    raw_pred_material: str
    confidence: float
    margin: float
    class_probs: dict[str, float]
    num_windows: int
    abstained: bool
    ready: bool
    decision_step: int | None
    aggregation: str = "mean_episode_logits"
    reason: str | None = None
    task_name: str | None = None
    gt_material: str | None = None
    contact_score_history: list[float] = field(default_factory=list)
    accepted_frame_indices: list[int] = field(default_factory=list)

    def to_dict(self, extra_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = asdict(self)
        if not self.contact_score_history:
            payload.pop("contact_score_history")
        if not self.accepted_frame_indices:
            payload.pop("accepted_frame_indices")
        if self.reason is None:
            payload.pop("reason")
        if self.task_name is None:
            payload.pop("task_name")
        if self.gt_material is None:
            payload.pop("gt_material")
        if extra_fields:
            payload.update(extra_fields)
        return payload


@dataclass(slots=True)
class MaterialStrategyProfile:
    strategy_name: str
    instruction: str
    policy_source: str
    post_contact_action_scale: float = 1.0
    stabilization_steps: int = 0
    max_action_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MaterialRoutingDecision:
    episode_id: str
    selected_material: str
    raw_material: str
    strategy_name: str
    instruction: str
    policy_source: str
    used_fallback: bool
    reason: str
    confidence: float
    margin: float
    ready: bool
    abstained: bool
    post_contact_action_scale: float = 1.0
    stabilization_steps: int = 0
    max_action_norm: float | None = None

    def to_dict(self, extra_fields: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = asdict(self)
        if extra_fields:
            payload.update(extra_fields)
        return payload


def build_prediction_from_logits(
    logits: Tensor,
    *,
    episode_id: str,
    num_windows: int,
    confidence_threshold: float,
    margin_threshold: float,
    min_windows_before_decision: int = 1,
    decision_step: int | None = None,
    aggregation: str = "mean_episode_logits",
    reason: str | None = None,
    task_name: str | None = None,
    gt_material: str | None = None,
    contact_score_history: list[float] | None = None,
    accepted_frame_indices: list[int] | None = None,
) -> MaterialProbePredictionResult:
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    top_k = torch.topk(probs, k=min(2, probs.numel()))
    pred_label = int(torch.argmax(probs).item())
    raw_pred_material = decode_material_label(pred_label)
    confidence = float(top_k.values[0].item())
    margin = (
        float(top_k.values[0].item() - top_k.values[1].item())
        if top_k.values.numel() > 1
        else confidence
    )
    ready = int(num_windows) >= int(min_windows_before_decision)
    abstained = (not ready) or confidence < float(confidence_threshold) or margin < float(margin_threshold)
    pred_material = raw_pred_material if not abstained else "unknown"
    class_probs = {
        material_name: float(probs[idx].item())
        for idx, material_name in enumerate(MATERIAL_CLASSES)
    }
    return MaterialProbePredictionResult(
        episode_id=str(episode_id),
        pred_material=pred_material,
        raw_pred_material=raw_pred_material,
        confidence=confidence,
        margin=margin,
        class_probs=class_probs,
        num_windows=int(num_windows),
        abstained=bool(abstained),
        ready=bool(ready),
        decision_step=decision_step,
        aggregation=aggregation,
        reason=reason,
        task_name=task_name,
        gt_material=gt_material,
        contact_score_history=list(contact_score_history or []),
        accepted_frame_indices=list(accepted_frame_indices or []),
    )


def build_empty_prediction(
    *,
    episode_id: str,
    reason: str = "no_contact_windows",
    aggregation: str = "mean_episode_logits",
) -> MaterialProbePredictionResult:
    return MaterialProbePredictionResult(
        episode_id=str(episode_id),
        pred_material="unknown",
        raw_pred_material="unknown",
        confidence=0.0,
        margin=0.0,
        class_probs={material_name: 0.0 for material_name in MATERIAL_CLASSES},
        num_windows=0,
        abstained=True,
        ready=False,
        decision_step=None,
        aggregation=aggregation,
        reason=reason,
    )


def average_logits_by_episode(
    logits: Tensor,
    labels: Tensor,
    episode_ids: list[str],
    task_names: list[str],
) -> tuple[Tensor, Tensor, list[str], list[str], list[int]]:
    sums: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    label_map: dict[str, Tensor] = {}
    task_map: dict[str, str] = {}
    ordered_ids: list[str] = []

    for idx, episode_id in enumerate(episode_ids):
        if episode_id not in sums:
            sums[episode_id] = logits[idx].clone()
            counts[episode_id] = 1
            label_map[episode_id] = labels[idx].clone()
            task_map[episode_id] = str(task_names[idx])
            ordered_ids.append(episode_id)
        else:
            sums[episode_id] += logits[idx]
            counts[episode_id] += 1

    mean_logits = torch.stack([sums[episode_id] / counts[episode_id] for episode_id in ordered_ids], dim=0)
    mean_labels = torch.stack([label_map[episode_id] for episode_id in ordered_ids], dim=0)
    mean_task_names = [task_map[episode_id] for episode_id in ordered_ids]
    num_windows = [counts[episode_id] for episode_id in ordered_ids]
    return mean_logits, mean_labels, ordered_ids, mean_task_names, num_windows


def coerce_prediction_result(prediction: MaterialProbePredictionResult | dict[str, Any]) -> MaterialProbePredictionResult:
    if isinstance(prediction, MaterialProbePredictionResult):
        return prediction
    return MaterialProbePredictionResult(
        episode_id=str(prediction.get("episode_id", "unknown_episode")),
        pred_material=str(prediction.get("pred_material", "unknown")),
        raw_pred_material=str(prediction.get("raw_pred_material", prediction.get("pred_material", "unknown"))),
        confidence=float(prediction.get("confidence", 0.0)),
        margin=float(prediction.get("margin", 0.0)),
        class_probs={
            material_name: float(prediction.get("class_probs", {}).get(material_name, 0.0))
            for material_name in MATERIAL_CLASSES
        },
        num_windows=int(prediction.get("num_windows", 0)),
        abstained=bool(prediction.get("abstained", True)),
        ready=bool(prediction.get("ready", False)),
        decision_step=prediction.get("decision_step"),
        aggregation=str(prediction.get("aggregation", "mean_episode_logits")),
        reason=prediction.get("reason"),
        task_name=prediction.get("task_name"),
        gt_material=prediction.get("gt_material"),
        contact_score_history=list(prediction.get("contact_score_history", [])),
        accepted_frame_indices=list(prediction.get("accepted_frame_indices", [])),
    )


def build_strategy_profile(
    raw_cfg: dict[str, Any],
    *,
    default_name: str,
    default_policy_source: str,
    default_instruction: str,
) -> MaterialStrategyProfile:
    instruction = str(raw_cfg.get("instruction", default_instruction)).strip()
    policy_source = str(raw_cfg.get("policy_source", default_policy_source)).strip()
    if not instruction:
        raise ValueError(f"Strategy profile {default_name!r} is missing an instruction.")
    if not policy_source:
        raise ValueError(f"Strategy profile {default_name!r} is missing a policy_source.")
    max_action_norm = raw_cfg.get("max_action_norm")
    return MaterialStrategyProfile(
        strategy_name=str(raw_cfg.get("strategy_name", default_name)),
        instruction=instruction,
        policy_source=policy_source,
        post_contact_action_scale=float(raw_cfg.get("post_contact_action_scale", 1.0)),
        stabilization_steps=int(raw_cfg.get("stabilization_steps", 0)),
        max_action_norm=None if max_action_norm is None else float(max_action_norm),
    )


def build_strategy_profiles(
    router_cfg: dict[str, Any],
) -> tuple[MaterialStrategyProfile, dict[str, MaterialStrategyProfile]]:
    policy_cfg = dict(router_cfg.get("policy", {}))
    fallback_cfg = dict(router_cfg.get("fallback", {}))
    default_policy_source = str(
        policy_cfg.get("default_source", fallback_cfg.get("policy_source", ""))
    ).strip()
    default_instruction = str(
        policy_cfg.get("default_instruction", fallback_cfg.get("instruction", "Lift the bag safely and steadily."))
    ).strip()
    fallback_profile = build_strategy_profile(
        fallback_cfg,
        default_name="default_safe",
        default_policy_source=default_policy_source,
        default_instruction=default_instruction,
    )
    material_profiles = {
        str(material_name): build_strategy_profile(
            dict(profile_cfg),
            default_name=str(material_name),
            default_policy_source=fallback_profile.policy_source,
            default_instruction=fallback_profile.instruction,
        )
        for material_name, profile_cfg in dict(router_cfg.get("materials", {})).items()
    }
    return fallback_profile, material_profiles


def select_material_strategy(
    prediction: MaterialProbePredictionResult | dict[str, Any],
    *,
    fallback_profile: MaterialStrategyProfile,
    material_profiles: dict[str, MaterialStrategyProfile],
) -> MaterialRoutingDecision:
    pred = coerce_prediction_result(prediction)
    profile = material_profiles.get(pred.pred_material)
    if pred.abstained:
        chosen_profile = fallback_profile
        used_fallback = True
        reason = "abstained_prediction"
    elif not pred.ready:
        chosen_profile = fallback_profile
        used_fallback = True
        reason = "prediction_not_ready"
    elif pred.pred_material == "unknown":
        chosen_profile = fallback_profile
        used_fallback = True
        reason = "unknown_prediction"
    elif profile is None:
        chosen_profile = fallback_profile
        used_fallback = True
        reason = "missing_material_profile"
    else:
        chosen_profile = profile
        used_fallback = False
        reason = "ready_prediction"

    return MaterialRoutingDecision(
        episode_id=pred.episode_id,
        selected_material=pred.pred_material,
        raw_material=pred.raw_pred_material,
        strategy_name=chosen_profile.strategy_name,
        instruction=chosen_profile.instruction,
        policy_source=chosen_profile.policy_source,
        used_fallback=used_fallback,
        reason=reason,
        confidence=pred.confidence,
        margin=pred.margin,
        ready=pred.ready,
        abstained=pred.abstained,
        post_contact_action_scale=chosen_profile.post_contact_action_scale,
        stabilization_steps=chosen_profile.stabilization_steps,
        max_action_norm=chosen_profile.max_action_norm,
    )


def postprocess_action(
    action: Tensor,
    profile: MaterialStrategyProfile | MaterialRoutingDecision,
    *,
    routed_step: int | None = None,
) -> Tensor:
    adjusted = action
    scale = float(profile.post_contact_action_scale)
    stabilization_steps = int(profile.stabilization_steps)
    if routed_step is not None and stabilization_steps > 0 and routed_step < stabilization_steps:
        ramp = float(routed_step + 1) / float(stabilization_steps)
        scale = min(scale, ramp)
    adjusted = adjusted * scale

    max_action_norm = profile.max_action_norm
    if max_action_norm is None:
        return adjusted

    if adjusted.ndim == 1:
        flat = adjusted.unsqueeze(0)
        squeezed = True
    else:
        flat = adjusted.reshape(adjusted.shape[0], -1)
        squeezed = False

    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    clip_scale = torch.clamp(float(max_action_norm) / safe_norms, max=1.0)
    clipped = flat * clip_scale
    if squeezed:
        return clipped.squeeze(0)
    return clipped.reshape_as(adjusted)