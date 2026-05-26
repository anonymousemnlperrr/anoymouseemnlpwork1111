"""B2 Contact-Local Semantic Actuation model.

This variant keeps the B2 concat path as the frozen control anchor and applies
physically-aware bottleneck semantics as a contact-local residual inside the
flow-matching velocity field instead of perturbing the whole action context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from VTLA.models.bottleneck import FixedQueryBottleneck, LanguageGuidedBottleneck
from VTLA.models.common import FlowMatchingActionHead, inject_lora, resolve_vlm_model_path
from VTLA.models.concat_fusion import ConcatFusion
from VTLA.models.lang_gtr import TactileLanguageAlignment
from VTLA.models.phys_classifier import PhysicalPropertyClassifier
from VTLA.models.tactile_encoder import DualTactileGridEncoder


CONTACT_FEATURE_DIM = 3


class B2ContactLocalSemanticActuationPolicy(nn.Module):
    """Frozen B2 anchor plus contact-local semantic residual velocity field."""

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        action_dim: int = 6,
        chunk_size: int = 1,
        n_queries: int = 16,
        d_bottleneck: int = 512,
        n_bottleneck_layers: int = 2,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_targets: Optional[list[str]] = None,
        freeze_vlm: bool = True,
        freeze_encoders: bool = True,
        disable_tactile: bool = False,
        disable_rgb: bool = False,
        bottleneck_variant: str = "lang_guided",
        residual_action_scale: float = 0.10,
        min_alpha: float = 0.0,
        max_alpha: float = 1.0,
        tactile_confidence_bias_init: float = -2.0,
        monotonic_margin: float = 0.02,
    ) -> None:
        super().__init__()
        self.vlm_model_id = vlm_model_id
        self.action_dim = int(action_dim)
        self.disable_tactile = bool(disable_tactile)
        self.disable_rgb = bool(disable_rgb)
        self.bottleneck_variant = bottleneck_variant
        self.residual_action_scale = float(residual_action_scale)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.monotonic_margin = float(monotonic_margin)

        if lora_targets is None:
            lora_targets = ["q_proj", "v_proj"]
        if bottleneck_variant not in {"lang_guided", "fixed_query"}:
            raise ValueError(f"Unsupported bottleneck_variant: {bottleneck_variant}")

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        resolved_vlm_path = resolve_vlm_model_path(vlm_model_id)
        local_files_only = Path(resolved_vlm_path).exists()
        self.resolved_vlm_path = resolved_vlm_path
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            resolved_vlm_path,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            resolved_vlm_path,
            local_files_only=local_files_only,
            use_fast=False,
        )
        d_vlm = int(self.vlm.config.hidden_size)
        self.d_vlm = d_vlm
        self.d_bottleneck = int(d_bottleneck)

        if freeze_vlm:
            self.vlm.requires_grad_(False)
        n_lora = inject_lora(self.vlm, lora_targets, rank=lora_rank, alpha=lora_alpha)
        print(f"[B2CLSA] Injected LoRA into {n_lora} layers")

        self.tactile_encoder = DualTactileGridEncoder(proj_dim=512)
        if freeze_encoders:
            self.tactile_encoder.requires_grad_(False)

        self.fusion = ConcatFusion(d_vlm=d_vlm, d_tac=512)
        self.action_head = FlowMatchingActionHead(
            hidden_size=d_vlm,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )

        bottleneck_cls = FixedQueryBottleneck if bottleneck_variant == "fixed_query" else LanguageGuidedBottleneck
        self.bottleneck = bottleneck_cls(
            d_vlm=d_vlm,
            d_bottleneck=d_bottleneck,
            n_queries=n_queries,
            n_layers=n_bottleneck_layers,
        )
        self.tla = TactileLanguageAlignment(d_bottleneck=d_bottleneck, d_vlm=d_vlm, proj_dim=256)
        self.phys_classifier = PhysicalPropertyClassifier(d_input=d_bottleneck)

        residual_input_dim = d_bottleneck + 512 + CONTACT_FEATURE_DIM
        self.residual_context_proj = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, d_vlm),
            nn.GELU(),
            nn.LayerNorm(d_vlm),
            nn.Linear(d_vlm, d_vlm),
        )
        self.tactile_confidence = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, d_bottleneck),
            nn.GELU(),
            nn.Linear(d_bottleneck, 1),
        )
        self.residual_action_head = FlowMatchingActionHead(
            hidden_size=d_vlm,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )

        nn.init.zeros_(self.residual_context_proj[-1].weight)
        nn.init.zeros_(self.residual_context_proj[-1].bias)
        nn.init.zeros_(self.tactile_confidence[-1].weight)
        nn.init.constant_(self.tactile_confidence[-1].bias, tactile_confidence_bias_init)
        nn.init.zeros_(self.residual_action_head.action_out_proj.weight)
        nn.init.zeros_(self.residual_action_head.action_out_proj.bias)

        vlm_dtype = next(self.vlm.parameters()).dtype
        for module in [
            self.tactile_encoder,
            self.fusion,
            self.action_head,
            self.bottleneck,
            self.tla,
            self.phys_classifier,
            self.residual_context_proj,
            self.tactile_confidence,
            self.residual_action_head,
        ]:
            module.to(vlm_dtype)

    def encode_vision_language(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if pixel_values is not None and not self.disable_rgb:
            kwargs["pixel_values"] = pixel_values.to(self.vlm.dtype)
            kwargs["image_grid_thw"] = image_grid_thw
        return self.vlm.model(**kwargs).last_hidden_state

    def encode_modalities(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tactile_grid: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        dtype = next(self.fusion.parameters()).dtype
        if self.disable_tactile:
            z_tactile = torch.zeros(tactile_grid.size(0), 512, device=tactile_grid.device, dtype=dtype)
        else:
            z_tactile, _ = self.tactile_encoder(tactile_grid.to(dtype=dtype))

        vlm_hidden = self.encode_vision_language(input_ids, attention_mask, pixel_values, image_grid_thw)
        seq_lens = attention_mask.sum(dim=1).long() - 1
        lang_cls = vlm_hidden[torch.arange(vlm_hidden.size(0), device=vlm_hidden.device), seq_lens]

        base_context = self.fusion(lang_cls, z_tactile)
        bottleneck_tokens = self.bottleneck(vlm_hidden, z_tactile)
        bn_cls = self.bottleneck.get_cls(bottleneck_tokens)
        return base_context, z_tactile, lang_cls, bn_cls, bottleneck_tokens

    def _normalize_contact_inputs(
        self,
        z_tactile: Tensor,
        residual_mask: Optional[Tensor],
        anchor_mask: Optional[Tensor],
        contact_features: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = z_tactile.size(0)
        device = z_tactile.device
        dtype = z_tactile.dtype

        if residual_mask is None:
            residual_mask = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            residual_mask = residual_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        if anchor_mask is None:
            anchor_mask = 1.0 - residual_mask
        else:
            anchor_mask = anchor_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        if contact_features is None:
            contact_features = torch.stack(
                [
                    residual_mask,
                    residual_mask,
                    torch.zeros_like(residual_mask),
                ],
                dim=-1,
            )
        else:
            contact_features = contact_features.to(device=device, dtype=dtype)
        return residual_mask, anchor_mask, contact_features

    def _build_residual_input(self, bn_cls: Tensor, z_tactile: Tensor, contact_features: Tensor) -> Tensor:
        return torch.cat([bn_cls, z_tactile, contact_features], dim=-1)

    def compute_alpha(
        self,
        z_tactile: Tensor,
        bn_cls: Tensor,
        residual_mask: Optional[Tensor] = None,
        anchor_mask: Optional[Tensor] = None,
        contact_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        residual_mask, anchor_mask, contact_features = self._normalize_contact_inputs(
            z_tactile,
            residual_mask,
            anchor_mask,
            contact_features,
        )
        residual_input = self._build_residual_input(bn_cls, z_tactile, contact_features)
        contact_gate = residual_mask.clamp(0.0, 1.0)
        tactile_conf = torch.sigmoid(self.tactile_confidence(residual_input).squeeze(-1))
        alpha = contact_gate * tactile_conf * self.residual_action_scale
        alpha = alpha.clamp(self.min_alpha, self.max_alpha).view(z_tactile.size(0), 1, 1)
        return alpha, residual_mask, anchor_mask, contact_features

    def build_residual_context(self, bn_cls: Tensor, z_tactile: Tensor, contact_features: Tensor) -> Tensor:
        residual_input = self._build_residual_input(bn_cls, z_tactile, contact_features)
        return self.residual_context_proj(residual_input)

    def _predict_velocity(self, head: FlowMatchingActionHead, context: Tensor, noisy_action: Tensor, time: Tensor) -> Tensor:
        action_emb = head.embed_action(noisy_action, time)
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, head.chunk_size, -1)
        return head.predict_velocity(action_emb + context)

    def _weighted_mean(self, values: Tensor, weights: Tensor) -> Tensor:
        weights = weights.to(device=values.device, dtype=values.dtype).clamp(min=0.0)
        denom = weights.sum()
        if float(denom.item()) <= 0.0:
            return values.mean()
        return (values * weights).sum() / denom

    def _compute_monotonic_loss(
        self,
        delta_action_norm: Tensor,
        task_names: Optional[list[str]],
        phase_names: Optional[list[str]],
        instruction_levels: Optional[list[str]],
    ) -> tuple[Tensor, Tensor]:
        device = delta_action_norm.device
        dtype = delta_action_norm.dtype
        if task_names is None or phase_names is None or instruction_levels is None:
            return torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)

        grouped: dict[tuple[str, str], dict[str, list[Tensor]]] = {}
        for delta_norm, task_name, phase_name, level in zip(
            delta_action_norm,
            task_names,
            phase_names,
            instruction_levels,
        ):
            if phase_name not in {"contact", "lift"}:
                continue
            level_name = str(level)
            if level_name not in {"L0", "L1", "L2"}:
                continue
            key = (str(task_name), str(phase_name))
            grouped.setdefault(key, {}).setdefault(level_name, []).append(delta_norm)

        loss_terms: list[Tensor] = []
        valid_groups = 0
        margin = torch.tensor(self.monotonic_margin, device=device, dtype=dtype)
        for level_map in grouped.values():
            if not all(level in level_map for level in ("L0", "L1", "L2")):
                continue
            d0 = torch.stack(level_map["L0"]).mean()
            d1 = torch.stack(level_map["L1"]).mean()
            d2 = torch.stack(level_map["L2"]).mean()
            loss_terms.append(F.relu(margin - (d1 - d0)))
            loss_terms.append(F.relu(margin - (d2 - d1)))
            valid_groups += 1

        if not loss_terms:
            return torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)
        return torch.stack(loss_terms).mean(), torch.tensor(float(valid_groups), device=device, dtype=dtype)

    def _compute_flow_terms(
        self,
        base_context: Tensor,
        residual_context: Tensor,
        alpha: Tensor,
        actions: Tensor,
        residual_mask: Tensor,
        anchor_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch_size = actions.shape[0]
        device = actions.device
        dtype = base_context.dtype
        actions = actions.to(dtype=dtype)

        noise = torch.randn_like(actions)
        time = torch.rand(batch_size, device=device, dtype=dtype)
        time_exp = time[:, None, None]
        x_t = time_exp * noise + (1 - time_exp) * actions
        u_t = noise - actions

        base_v = self._predict_velocity(self.action_head, base_context, x_t, time)
        residual_v = self._predict_velocity(self.residual_action_head, residual_context, x_t, time)
        correction_v = alpha.to(dtype=dtype) * residual_v
        final_v = base_v + correction_v

        per_sample_action_mse = (final_v - u_t).pow(2).mean(dim=(1, 2))
        per_sample_base_action_mse = (base_v - u_t).pow(2).mean(dim=(1, 2))
        per_sample_anchor_mse = (final_v - base_v.detach()).pow(2).mean(dim=(1, 2))
        per_sample_delta_norm = correction_v.pow(2).mean(dim=(1, 2)).sqrt()

        contact_action_loss = self._weighted_mean(per_sample_action_mse, residual_mask)
        overall_action_loss = per_sample_action_mse.mean()
        noncontact_anchor_loss = self._weighted_mean(per_sample_anchor_mse, anchor_mask)
        residual_norm_loss = correction_v.pow(2).mean()
        base_action_loss = per_sample_base_action_mse.mean()
        contact_base_action_loss = self._weighted_mean(per_sample_base_action_mse, residual_mask)

        return {
            "contact_action_loss": contact_action_loss,
            "overall_action_loss": overall_action_loss,
            "noncontact_anchor_loss": noncontact_anchor_loss,
            "residual_norm_loss": residual_norm_loss,
            "base_action_loss": base_action_loss,
            "contact_base_action_loss": contact_base_action_loss,
            "per_sample_action_mse": per_sample_action_mse.detach(),
            "per_sample_base_action_mse": per_sample_base_action_mse.detach(),
            "per_sample_delta_norm": per_sample_delta_norm.detach(),
            "alpha_mean": alpha.mean().detach(),
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tactile_grid: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        task_names: Optional[list[str]] = None,
        residual_mask: Optional[Tensor] = None,
        anchor_mask: Optional[Tensor] = None,
        contact_features: Optional[Tensor] = None,
        phase_names: Optional[list[str]] = None,
        instruction_levels: Optional[list[str]] = None,
    ) -> dict[str, Tensor]:
        results: dict[str, Tensor] = {}
        dtype = next(self.fusion.parameters()).dtype
        base_context, z_tactile, lang_cls, bn_cls, bottleneck_tokens = self.encode_modalities(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        alpha, residual_mask, anchor_mask, contact_features = self.compute_alpha(
            z_tactile,
            bn_cls,
            residual_mask=residual_mask,
            anchor_mask=anchor_mask,
            contact_features=contact_features,
        )
        residual_context = self.build_residual_context(bn_cls, z_tactile, contact_features)

        results["base_context"] = base_context
        results["bottleneck_tokens"] = bottleneck_tokens
        results["residual_context"] = residual_context
        results["alpha"] = alpha.squeeze(-1).squeeze(-1)
        results["alpha_mean"] = alpha.mean().detach()
        results["residual_mask_mean"] = residual_mask.mean().detach()
        results["anchor_mask_mean"] = anchor_mask.mean().detach()
        results["contact_prior_mean"] = contact_features[:, 1].mean().detach()
        results["metadata_mask_mean"] = contact_features[:, 2].mean().detach()

        if task_names is not None:
            results["align_loss"] = self.tla.compute_loss(bn_cls, lang_cls, task_names)
            phys_logits = self.phys_classifier(bn_cls)
            results["phys_logits"] = phys_logits
            from VTLA.models.phys_classifier import get_phys_labels

            phys_labels = get_phys_labels(task_names, phys_logits.device)
            results["phys_loss"] = self.phys_classifier.compute_loss(phys_logits, phys_labels)
        else:
            results["align_loss"] = torch.zeros((), device=tactile_grid.device, dtype=dtype)
            results["phys_loss"] = torch.zeros((), device=tactile_grid.device, dtype=dtype)

        if actions is not None:
            flow_terms = self._compute_flow_terms(
                base_context,
                residual_context,
                alpha,
                actions,
                residual_mask,
                anchor_mask,
            )
            results.update(flow_terms)
            monotonic_loss, monotonic_groups = self._compute_monotonic_loss(
                flow_terms["per_sample_delta_norm"],
                task_names,
                phase_names,
                instruction_levels,
            )
            results["monotonic_loss"] = monotonic_loss
            results["monotonic_group_count"] = monotonic_groups.detach()
            results["action_loss"] = flow_terms["contact_action_loss"]

        return results

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tactile_grid: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        n_steps: int = 10,
        residual_mask: Optional[Tensor] = None,
        contact_features: Optional[Tensor] = None,
        anchor_mask: Optional[Tensor] = None,
    ) -> Tensor:
        base_context, z_tactile, _, bn_cls, _ = self.encode_modalities(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        alpha, _, _, contact_features = self.compute_alpha(
            z_tactile,
            bn_cls,
            residual_mask=residual_mask,
            anchor_mask=anchor_mask,
            contact_features=contact_features,
        )
        residual_context = self.build_residual_context(bn_cls, z_tactile, contact_features)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = base_context.dtype
        x = torch.randn(batch_size, self.action_head.chunk_size, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            time = torch.full((batch_size,), step / n_steps, device=device, dtype=dtype)
            base_v = self._predict_velocity(self.action_head, base_context, x, time)
            residual_v = self._predict_velocity(self.residual_action_head, residual_context, x, time)
            x = x + (base_v + alpha.to(dtype=dtype) * residual_v) * dt
        return x

    def load_stage_a_components(self, ckpt_path: str, load_tactile_encoder: bool = False) -> tuple[int, int]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        prefixes = ["bottleneck.", "tla.", "phys_classifier."]
        if load_tactile_encoder:
            prefixes.append("tactile_encoder.")
        filtered = {key: value for key, value in state.items() if any(key.startswith(prefix) for prefix in prefixes)}
        msg = self.load_state_dict(filtered, strict=False)
        return len(filtered), len(msg.missing_keys)

    def load_b2_anchor(self, ckpt_path: str) -> tuple[int, int]:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        prefixes = ("tactile_encoder.", "fusion.", "action_head.")
        filtered = {
            key: value
            for key, value in state.items()
            if key.startswith(prefixes) or "lora_" in key
        }
        msg = self.load_state_dict(filtered, strict=False)
        return len(filtered), len(msg.missing_keys)

    def freeze_b2_anchor(
        self,
        freeze_lora: bool = True,
        freeze_fusion: bool = True,
        freeze_action_head: bool = True,
        freeze_tactile_encoder: bool = True,
    ) -> None:
        if freeze_fusion:
            self.fusion.requires_grad_(False)
        if freeze_action_head:
            self.action_head.requires_grad_(False)
        if freeze_tactile_encoder:
            self.tactile_encoder.requires_grad_(False)
        if freeze_lora:
            for name, param in self.vlm.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(False)

    def get_trainable_params(self, stage: str = "B") -> list[dict]:
        if stage != "B":
            raise ValueError(f"Unknown stage: {stage}")
        groups = []
        specs = [
            (self.residual_context_proj, 1e-4, "residual_context_proj"),
            (self.tactile_confidence, 1e-4, "tactile_confidence"),
            (self.residual_action_head, 1e-4, "residual_action_head"),
            (self.bottleneck, 5e-5, "bottleneck"),
            (self.tla, 5e-5, "tla"),
            (self.phys_classifier, 5e-5, "phys_classifier"),
            (self.tactile_encoder, 1e-5, "tactile_encoder"),
            (self.fusion, 1e-5, "fusion"),
            (self.action_head, 1e-5, "action_head"),
        ]
        lora_params = [param for _, param in self.vlm.named_parameters() if param.requires_grad]
        if lora_params:
            groups.append({"params": lora_params, "lr": 1e-5, "name": "lora"})
        for module, lr, name in specs:
            params = [param for param in module.parameters() if param.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr, "name": name})
        return groups

    def save_checkpoint(self, path: str, stage: str, epoch: int, optimizer=None, extra_state: Optional[dict] = None):
        prefixes = [
            "tactile_encoder.",
            "fusion.",
            "action_head.",
            "bottleneck.",
            "tla.",
            "phys_classifier.",
            "residual_context_proj.",
            "tactile_confidence.",
            "residual_action_head.",
        ]
        trainable_state = {
            key: value
            for key, value in self.state_dict().items()
            if any(key.startswith(prefix) for prefix in prefixes) or "lora_" in key
        }
        payload = {
            "stage": stage,
            "epoch": epoch,
            "model": trainable_state,
            "model_type": "b2_contact_local_semantic_actuation",
            "vlm_model_id": self.vlm_model_id,
            "action_dim": self.action_dim,
            "chunk_size": self.action_head.chunk_size,
            "disable_tactile": self.disable_tactile,
            "disable_rgb": self.disable_rgb,
            "bottleneck_variant": self.bottleneck_variant,
            "residual_action_scale": self.residual_action_scale,
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
            "monotonic_margin": self.monotonic_margin,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if extra_state:
            payload.update(extra_state)
        torch.save(payload, path)
        print(f"[B2CLSA] Saved checkpoint: {path} ({len(trainable_state)} params)")