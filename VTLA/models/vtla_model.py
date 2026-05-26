"""
VTLA/models/vtla_model.py

VTLA (Vision-Tactile-Language-Action) 完整模型

架构:
  Qwen2-VL-2B (frozen backbone + LoRA)
  + DualTactileGridEncoder (Stage 3 pretrained)
  + LanguageGuidedBottleneck (K=16, d=512, 2 layers) ← C1: 核心架构创新
  + TactileLanguageAlignment (对比对齐)             ← C2: 训练目标创新
  + PhysicalPropertyClassifier (2 binary heads)      ← 辅助任务 (compliance, fragility)
  + TactileMapHead (可视化辅助, 不参与主 loss)
  + Flow Matching Action Head (6-DoF)

Stage A: 训练 Bottleneck + TLA + PhysProp, 冻结 VLM + encoders
Stage B: 解冻 LoRA + Action Head, 端到端微调

论文贡献:
  C1 = Language-Guided Cross-Modal Bottleneck (架构)
  C2 = Tactile-Language Semantic Alignment (训练目标, 不是"重建")
  C3 = Instruction Gradient / Minimal-pair (验证方法)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from VTLA.models.tactile_encoder import DualTactileGridEncoder
from VTLA.models.bottleneck import LanguageGuidedBottleneck, FixedQueryBottleneck
from VTLA.models.lang_gtr import TactileLanguageAlignment, TactileMapHead
from VTLA.models.material_probe_content_classifier import MaterialProbeContentClassifier
from VTLA.models.common import FlowMatchingActionHead, inject_lora, resolve_vlm_model_path
from VTLA.models.phys_classifier import PhysicalPropertyClassifier


# ============================================================================
# VTLA Model
# ============================================================================

class VTLAModel(nn.Module):
    """
    Vision-Tactile-Language-Action Model

    Qwen2-VL-2B backbone + Language-Guided Bottleneck + Action Head

    forward() 返回 dict:
      - "action_loss":  Flow Matching loss (Stage B)
      - "align_loss":   Tactile-Language Alignment loss (C2)
      - "phys_loss":    Physical Property BCE loss
      - "bottleneck_tokens": [B, K, d_vlm]
      - "pred_tactile":  [B, 16, 16] (可视化, no_grad)
    """

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
        freeze_encoders: bool = False,
        disable_tactile: bool = False,
        disable_rgb: bool = False,
        use_dummy_vlm: bool = False,
        bottleneck_variant: str = "lang_guided",
    ):
        super().__init__()
        self.vlm_model_id = vlm_model_id
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_queries = n_queries
        self.d_bottleneck = d_bottleneck
        self.n_bottleneck_layers = n_bottleneck_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.disable_tactile = disable_tactile
        self.disable_rgb = disable_rgb
        if bottleneck_variant not in {"lang_guided", "fixed_query"}:
            raise ValueError(
                f"Unsupported bottleneck_variant: {bottleneck_variant!r}. "
                "Expected one of: 'lang_guided', 'fixed_query'."
            )
        self.bottleneck_variant = bottleneck_variant

        if lora_targets is None:
            lora_targets = ["q_proj", "v_proj"]

        # ---- VLM backbone ----
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
        d_vlm = self.vlm.config.hidden_size  # 1536 for Qwen2-VL-2B

        if freeze_vlm:
            self.vlm.requires_grad_(False)

        # Inject LoRA
        n_lora = inject_lora(self.vlm, lora_targets, rank=lora_rank, alpha=lora_alpha)
        print(f"[VTLA] Injected LoRA into {n_lora} layers (rank={lora_rank})")

        # ---- Tactile Encoder ----
        self.tactile_encoder = DualTactileGridEncoder(proj_dim=512)

        if freeze_encoders:
            self.tactile_encoder.requires_grad_(False)

        # ---- Bottleneck (语言引导 / 固定查询 二选一; A1 消融) ----
        bottleneck_cls = (
            FixedQueryBottleneck
            if bottleneck_variant == "fixed_query"
            else LanguageGuidedBottleneck
        )
        self.bottleneck = bottleneck_cls(
            d_vlm=d_vlm,
            d_bottleneck=d_bottleneck,
            n_queries=n_queries,
            n_layers=n_bottleneck_layers,
        )

        # ---- Tactile-Language Alignment (C2: 训练目标) ----
        self.tla = TactileLanguageAlignment(
            d_bottleneck=d_bottleneck,
            d_vlm=d_vlm,
            proj_dim=256,
        )

        # ---- Tactile Map Head (可视化辅助, 不参与主 loss) ----
        self.tactile_map_head = TactileMapHead(
            vision_dim=512,
            lang_dim=d_vlm,
        )

        # ---- Physical Property Classifier (辅助任务) ----
        self.phys_classifier = PhysicalPropertyClassifier(d_input=d_bottleneck)

        # ---- Black-Bag Content Classifier (辅助语义读出) ----
        self.material_probe_classifier = MaterialProbeContentClassifier(d_input=d_bottleneck)

        # ---- Flow Matching Action Head ----
        self.action_head = FlowMatchingActionHead(
            hidden_size=d_vlm,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )

        # ---- dtype: match custom modules to VLM dtype (bf16) ----
        vlm_dtype = next(self.vlm.parameters()).dtype
        for mod in [self.bottleneck, self.tla, self.phys_classifier, self.material_probe_classifier,
                     self.tactile_map_head, self.action_head, self.tactile_encoder]:
            mod.to(vlm_dtype)

    def compute_bottleneck(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tactile_grid: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        return_attentions: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Optional[list[dict[str, Tensor]]]]:
        """Encode modalities and run the language-guided bottleneck once."""
        dtype = next(self.bottleneck.parameters()).dtype

        if self.disable_tactile:
            z_tactile = torch.zeros(
                tactile_grid.size(0), 512, device=tactile_grid.device, dtype=dtype
            )
        else:
            z_tactile, _ = self.tactile_encoder(tactile_grid.to(dtype=dtype))

        vlm_hidden = self.encode_vision_language(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )
        if return_attentions:
            bottleneck_tokens, attention_maps = self.bottleneck(
                vlm_hidden, z_tactile, return_attentions=True,
            )
        else:
            bottleneck_tokens = self.bottleneck(vlm_hidden, z_tactile)
            attention_maps = None

        return bottleneck_tokens, z_tactile, vlm_hidden, attention_maps

    def encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        获取 VLM 语言 hidden states (text-only, 无 RGB)
        input_ids:      [B, L]
        attention_mask:  [B, L]
        returns: lang_hidden [B, L, d_vlm]
        """
        outputs = self.vlm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.last_hidden_state  # [B, L, d_vlm]

    def encode_vision_language(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> Tensor:
        """
        获取 VLM vision+language hidden states (3-modality 中的 RGB+Language 分支)

        当 pixel_values 为 None 时退化为 encode_language (text-only)
        当 pixel_values 不为 None 时, Qwen2-VL 的 SigLIP ViT 编码 RGB,
        返回的 hidden states 同时包含 vision + language 信息

        input_ids:       [B, L]
        attention_mask:   [B, L]
        pixel_values:     [B, N_patches, C, pH, pW] (Qwen2-VL 格式)
        image_grid_thw:   [B, N_images, 3] (T, H, W grid)
        returns: vlm_hidden [B, L', d_vlm]  (L' >= L due to vision tokens)
        """
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if pixel_values is not None and not self.disable_rgb:
            kwargs["pixel_values"] = pixel_values.to(self.vlm.dtype)
            kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.vlm.model(**kwargs)
        return outputs.last_hidden_state  # [B, L', d_vlm]

    def forward(
        self,
        input_ids: Tensor,           # [B, L] tokenized instruction
        attention_mask: Tensor,       # [B, L]
        tactile_grid: Tensor,         # [B, T, 2, 16, 16]
        pixel_values: Optional[Tensor] = None,      # [B, N, C, pH, pW] Qwen2-VL RGB
        image_grid_thw: Optional[Tensor] = None,     # [B, N_img, 3]
        actions: Optional[Tensor] = None,  # [B, chunk_size, action_dim]
        task_names: Optional[list[str]] = None,  # for phys labels
        material_labels: Optional[Tensor] = None,
        is_material_probe: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """完整前向传播 (3-modality: RGB + Tactile + Language → Action)"""
        results = {}
        device = input_ids.device
        dtype = next(self.bottleneck.parameters()).dtype

        # 1. 编码各模态 + Bottleneck
        bottleneck_tokens, z_tactile, vlm_hidden, _ = self.compute_bottleneck(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_attentions=False,
        )

        # 语言 CLS: 取最后一个非 padding token
        seq_lens = attention_mask.sum(dim=1).long() - 1     # [B]
        lang_cls = vlm_hidden[torch.arange(vlm_hidden.size(0)), seq_lens]  # [B, d_vlm]

        results["bottleneck_tokens"] = bottleneck_tokens

        # 3. Tactile-Language Alignment (C2: 物理语义对齐)
        # 挂在 bottleneck 之后, 梯度直接强化 bottleneck 的语义编码
        bn_cls = self.bottleneck.get_cls(bottleneck_tokens)  # [B, d_bn]
        results["bottleneck_cls"] = bn_cls
        if task_names is not None:
            results["align_loss"] = self.tla.compute_loss(bn_cls, lang_cls, task_names)
        else:
            results["align_loss"] = torch.tensor(0.0, device=tactile_grid.device)

        # 可视化头 (不产生主 loss, 仅记录 pred_tactile 供 debug/可视化)
        with torch.no_grad():
            pred_tactile = self.tactile_map_head(z_tactile, lang_cls)
        results["pred_tactile"] = pred_tactile

        # 4. Physical Property Classifier (辅助)
        phys_logits = self.phys_classifier(bn_cls)
        results["phys_logits"] = phys_logits
        if task_names is not None:
            from VTLA.models.phys_classifier import get_phys_labels
            phys_labels = get_phys_labels(task_names, phys_logits.device)
            results["phys_loss"] = self.phys_classifier.compute_loss(phys_logits, phys_labels)

        # 5. Black-Bag Content Classifier (辅助语义读出)
        material_probe_logits = self.material_probe_classifier(bn_cls)
        results["material_probe_logits"] = material_probe_logits
        if material_labels is not None:
            labels = material_labels.to(device=material_probe_logits.device).long()
            valid_mask = labels >= 0
            if is_material_probe is not None:
                valid_mask = valid_mask & is_material_probe.to(device=material_probe_logits.device).bool()
            results["material_probe_num_valid"] = valid_mask.sum()
            if valid_mask.any():
                results["material_probe_loss"] = self.material_probe_classifier.compute_loss(
                    material_probe_logits[valid_mask],
                    labels[valid_mask],
                )
            else:
                results["material_probe_loss"] = material_probe_logits.sum() * 0.0

        # 6. Action Head (Stage B)
        if actions is not None:
            # 使用 bottleneck tokens 的 mean 作为 action context
            action_context = bottleneck_tokens.mean(dim=1)  # [B, d_vlm]
            results["action_loss"] = self.action_head.compute_loss(
                action_context, actions.to(dtype=dtype)
            )

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
    ) -> Tensor:
        """推理: 生成动作 (3-modality)"""
        bottleneck_tokens, _, _, _ = self.compute_bottleneck(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_attentions=False,
        )
        action_context = bottleneck_tokens.mean(dim=1)
        return self.action_head.sample(action_context, n_steps=n_steps)

    @torch.no_grad()
    def predict_material_probe_content(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        tactile_grid: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        bottleneck_tokens, _, _, _ = self.compute_bottleneck(
            input_ids,
            attention_mask,
            tactile_grid,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_attentions=False,
        )
        bn_cls = self.bottleneck.get_cls(bottleneck_tokens)
        logits = self.material_probe_classifier(bn_cls)
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        pred_labels = probs.argmax(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred_labels": pred_labels,
            "bottleneck_cls": bn_cls,
        }

    def get_trainable_params(self, stage: str = "A") -> list[dict]:
        """
        返回分组的可训练参数

        Stage A: Bottleneck + TLA + PhysProp (encoders optional)
        Stage B: LoRA + Action Head + Bottleneck + TLA (fine-tune)
        """
        if stage == "A":
            return [
                {"params": self.bottleneck.parameters(), "lr": 1e-4, "name": "bottleneck"},
                {"params": self.tla.parameters(), "lr": 1e-4, "name": "tla"},
                {"params": self.phys_classifier.parameters(), "lr": 1e-3, "name": "phys_classifier"},
                {"params": self.tactile_encoder.parameters(), "lr": 1e-5, "name": "tactile_encoder"},
            ]
        elif stage in {"A_material_probe", "material_probe"}:
            return [
                {"params": self.material_probe_classifier.parameters(), "lr": 1e-3, "name": "material_probe_classifier"},
                {"params": self.bottleneck.parameters(), "lr": 5e-5, "name": "bottleneck"},
                {"params": self.phys_classifier.parameters(), "lr": 5e-5, "name": "phys_classifier"},
                {"params": self.tactile_encoder.parameters(), "lr": 1e-5, "name": "tactile_encoder"},
            ]
        elif stage == "B":
            lora_params = [p for n, p in self.vlm.named_parameters() if p.requires_grad]
            return [
                {"params": lora_params, "lr": 2e-5, "name": "lora"},
                {"params": self.action_head.parameters(), "lr": 1e-4, "name": "action_head"},
                {"params": self.bottleneck.parameters(), "lr": 5e-5, "name": "bottleneck"},
                {"params": self.tla.parameters(), "lr": 5e-5, "name": "tla"},
            ]
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def load_stage3_weights(self, ckpt_path: str):
        """加载 Stage 3 预训练权重到 tactile encoder"""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)

        # Tactile encoder
        tac_weights = {}
        for k, v in state.items():
            if k.startswith("tactile_encoder."):
                new_k = k.replace("tactile_encoder.", "encoder.", 1)
                tac_weights[new_k] = v
        if tac_weights:
            msg = self.tactile_encoder.load_state_dict(tac_weights, strict=False)
            print(f"[VTLA] Loaded {len(tac_weights)} tactile encoder params. Missing: {len(msg.missing_keys)}")

    def save_checkpoint(self, path: str, stage: str, epoch: int, optimizer=None, extra_state: Optional[dict] = None):
        """保存 checkpoint (只保存可训练参数)"""
        trainable_state = {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(prefix) for prefix in [
                "bottleneck.", "tla.", "phys_classifier.", "material_probe_classifier.", "action_head.",
                "tactile_encoder.", "tactile_map_head.",
            ]) or "lora_" in k
        }
        ckpt = {
            "stage": stage,
            "epoch": epoch,
            "model": trainable_state,
            "vlm_model_id": self.vlm_model_id,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "n_queries": self.n_queries,
            "d_bottleneck": self.d_bottleneck,
            "n_bottleneck_layers": self.n_bottleneck_layers,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "bottleneck_variant": self.bottleneck_variant,
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        if extra_state:
            ckpt.update(extra_state)
        torch.save(ckpt, path)
        print(f"[VTLA] Saved checkpoint: {path} ({len(trainable_state)} params)")

    def load_checkpoint(self, path: str, strict: bool = False) -> dict:
        """加载 VTLA checkpoint, 返回原始 checkpoint 字典。"""
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        msg = self.load_state_dict(state, strict=strict)
        if not strict:
            print(
                f"[VTLA] Loaded checkpoint: {path}. "
                f"Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
            )
        else:
            print(f"[VTLA] Loaded checkpoint: {path}")
        return ckpt
