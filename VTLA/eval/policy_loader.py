from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch


def _validate_policy_object(policy: Any) -> None:
    if not hasattr(policy, "predict_action"):
        raise AttributeError("Policy object must implement predict_action(...).")

    has_tokenizer = hasattr(policy, "tokenizer")
    has_processor = hasattr(policy, "processor")
    processor_has_tokenizer = has_processor and getattr(policy.processor, "tokenizer", None) is not None
    if not (has_tokenizer or has_processor or processor_has_tokenizer):
        raise AttributeError(
            "Policy object must expose tokenizer, processor, or processor.tokenizer."
        )


def get_tokenizer_or_processor(policy: Any) -> Any:
    processor = getattr(policy, "processor", None)
    if processor is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    tokenizer = getattr(policy, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    if processor is not None:
        return processor
    raise AttributeError("Could not resolve tokenizer/processor from policy object.")


def get_rgb_processor(policy: Any) -> Any | None:
    return getattr(policy, "processor", None)


def inspect_policy_source(policy_source: str) -> dict[str, Any]:
    path = Path(policy_source)
    if path.suffix.lower() == ".json":
        spec = json.loads(path.read_text(encoding="utf-8"))
        return {
            "source_type": "external_spec",
            "policy_type": spec.get("policy_type", "external_python"),
            "label": spec.get("label", path.stem),
            "checkpoint_epoch": spec.get("checkpoint_epoch"),
            "load_rgb": bool(spec.get("load_rgb", True)),
            "supports_attention_kl": bool(spec.get("supports_attention_kl", False)),
            "spec": spec,
        }

    checkpoint = torch.load(policy_source, map_location="cpu", weights_only=False)
    return {
        "source_type": "vtla_checkpoint",
        "policy_type": checkpoint.get("model_type", "vtla"),
        "label": checkpoint.get("label", path.stem),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "load_rgb": bool(not checkpoint.get("disable_rgb", False)),
        "supports_attention_kl": bool(checkpoint.get("model_type") != "concat"),
        "checkpoint": checkpoint,
    }


def _load_vtla_policy_from_checkpoint(policy_source: str, metadata: dict[str, Any]):
    checkpoint = metadata.get("checkpoint")
    if checkpoint is None:
        checkpoint = torch.load(policy_source, map_location="cpu", weights_only=False)
        metadata["checkpoint"] = checkpoint

    model_type = checkpoint.get("model_type")
    if model_type == "b2_contact_local_semantic_actuation":
        from VTLA.eval.contact_local_policy_bridge import build_policy

        policy, extra_metadata = build_policy(policy_source)
        merged_metadata = dict(metadata)
        if isinstance(extra_metadata, dict):
            merged_metadata.update(extra_metadata)
        return policy, merged_metadata

    from VTLA.models.vtla_model import VTLAModel

    disable_rgb = bool(checkpoint.get("disable_rgb", False))
    policy = VTLAModel(
        disable_tactile=bool(checkpoint.get("disable_tactile", False)),
        disable_rgb=disable_rgb,
        bottleneck_variant=str(checkpoint.get("bottleneck_variant", "lang_guided")),
    )
    policy.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    return policy, metadata


def _load_external_python_policy(spec: dict[str, Any], metadata: dict[str, Any]):
    module_name = spec["module"]
    factory_name = spec.get("factory", "build_policy")
    builder_kwargs = spec.get("builder_kwargs", {})

    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    built = factory(**builder_kwargs)

    if isinstance(built, tuple) and len(built) == 2:
        policy, extra_metadata = built
        if isinstance(extra_metadata, dict):
            metadata.update(extra_metadata)
    else:
        policy = built

    _validate_policy_object(policy)
    return policy, metadata


def load_policy_from_source(policy_source: str):
    metadata = inspect_policy_source(policy_source)
    if metadata["source_type"] == "external_spec":
        spec = metadata["spec"]
        policy_type = spec.get("policy_type", "external_python")
        if policy_type == "external_python":
            return _load_external_python_policy(spec, metadata)
        raise ValueError(f"Unsupported external policy type: {policy_type}")

    return _load_vtla_policy_from_checkpoint(policy_source, metadata)
