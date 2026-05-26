"""
VTLA/models/bottleneck.py

Language-Guided Cross-Modal Bottleneck (核心创新模块)

设计: Q-Former 风格的信息瓶颈
  - K=16 个可学习 query token
  - 先用 language cross-attention 对 query 进行语义调制
  - 再用 perception cross-attention 从多模态特征中提取信息
  - 输出 [B, K, d_vlm] 注入 VLM prefix

参数量: ~10M (2层 × 2个cross-attn × d=512 × 8heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BottleneckLayer(nn.Module):
    """单层 Bottleneck: Language CrossAttn → Perception CrossAttn → FFN"""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Language cross-attention: query attends to language
        self.lang_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.lang_norm = nn.LayerNorm(d_model)

        # Perception cross-attention: language-conditioned query attends to perception
        self.perc_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.perc_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: Tensor,
        lang_feats: Tensor,
        perc_feats: Tensor,
        return_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """
        queries:    [B, K, d_model]   可学习 query
        lang_feats: [B, L, d_model]   语言特征序列
        perc_feats: [B, N, d_model]   感知特征 (tactile)
        returns:    [B, K, d_model]   调制后的 query
        """
        # Step 1: Language conditioning
        q = self.lang_norm(queries)
        attn_out, lang_attn = self.lang_cross_attn(
            q, lang_feats, lang_feats,
            need_weights=return_attentions,
            average_attn_weights=False,
        )
        queries = queries + attn_out

        # Step 2: Perception extraction
        q = self.perc_norm(queries)
        attn_out, perc_attn = self.perc_cross_attn(
            q, perc_feats, perc_feats,
            need_weights=return_attentions,
            average_attn_weights=False,
        )
        queries = queries + attn_out

        # Step 3: FFN
        queries = queries + self.ffn(self.ffn_norm(queries))
        if return_attentions:
            return queries, {"lang": lang_attn, "perc": perc_attn}
        return queries


class LanguageGuidedBottleneck(nn.Module):
    """
    Language-Guided Cross-Modal Bottleneck

    完整数据流:
      1. 各模态特征先投影到 d_bottleneck=512:
         - tactile [B,512] → [B,1,512]
         - (VLM image tokens 在 vtla_model.py 中处理)
      2. K=16 个 learned query 经过 N_layers 层 BottleneckLayer:
         - 先 cross-attend to language → 语义调制
         - 再 cross-attend to perception → 信息提取
      3. 输出 [B, K, d_vlm] 注入 VLM hidden space

    输入:
      lang_hidden:  [B, L, d_vlm]     VLM 语言隐层 (Qwen2-VL-2B: d=1536)
    z_tactile:    [B, 512]          触觉全局特征

    输出:
      bottleneck_tokens: [B, K, d_vlm]  注入 VLM 的 prefix tokens
    """

    def __init__(
        self,
        d_vlm: int = 1536,        # Qwen2-VL-2B hidden size
        d_bottleneck: int = 512,   # 瓶颈维度
        n_queries: int = 16,       # learned query 数量
        n_layers: int = 2,         # BottleneckLayer 层数
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_bottleneck = d_bottleneck
        self.d_vlm = d_vlm

        # Learned queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_bottleneck) * 0.02)

        # 输入投影: 各模态 → d_bottleneck
        self.tactile_proj = nn.Linear(512, d_bottleneck)
        self.lang_proj = nn.Linear(d_vlm, d_bottleneck)

        # Bottleneck layers
        self.layers = nn.ModuleList([
            BottleneckLayer(d_model=d_bottleneck, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 输出投影: d_bottleneck → d_vlm
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_bottleneck),
            nn.Linear(d_bottleneck, d_vlm),
        )

        # CLS 投影: d_vlm → d_bottleneck (用于 PhysClassifier)
        self.cls_proj = nn.Linear(d_vlm, d_bottleneck)

    def forward(
        self,
        lang_hidden: Tensor,    # [B, L, d_vlm]
        z_tactile: Tensor,      # [B, 512]
        return_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, list[dict[str, Tensor]]]:
        B = lang_hidden.shape[0]

        # 投影到 bottleneck 空间
        lang_feats = self.lang_proj(lang_hidden)                          # [B, L, d_bn]
        tac_feats = self.tactile_proj(z_tactile).unsqueeze(1)             # [B, 1, d_bn]

        # 触觉感知特征池
        perc_feats = tac_feats                                            # [B, 1, d_bn]

        # Bottleneck 层
        queries = self.queries.expand(B, -1, -1)                         # [B, K, d_bn]
        attention_maps = []
        for layer in self.layers:
            if return_attentions:
                queries, layer_attn = layer(
                    queries, lang_feats, perc_feats, return_attentions=True,
                )
                attention_maps.append(layer_attn)
            else:
                queries = layer(queries, lang_feats, perc_feats)

        # 投影回 VLM 空间
        bottleneck_tokens = self.output_proj(queries)                    # [B, K, d_vlm]
        if return_attentions:
            return bottleneck_tokens, attention_maps
        return bottleneck_tokens

    def get_cls(self, bottleneck_tokens: Tensor) -> Tensor:
        """从 bottleneck 输出中提取 CLS token 并投影回 d_bottleneck.

        Args:
            bottleneck_tokens: [B, K, d_vlm] — 经 output_proj 后的 VLM 维度
        Returns:
            cls_token: [B, d_bottleneck]
        """
        return self.cls_proj(bottleneck_tokens[:, 0])  # [B, d_vlm] → [B, d_bn]


# ============================================================================
# Ablation variants
# ============================================================================


class PerceptionOnlyLayer(nn.Module):
    """跳过 language cross-attention, 仅做 perception cross-attention + FFN."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.perc_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.perc_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, queries: Tensor, _lang_feats: Tensor, perc_feats: Tensor) -> Tensor:
        q = self.perc_norm(queries)
        attn_out, _ = self.perc_cross_attn(q, perc_feats, perc_feats)
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class FixedQueryBottleneck(nn.Module):
    """
    Ablation A1: Fixed-Query Bottleneck (no language cross-attention).

    Queries only attend to perception features (tactile),
    without language-guided conditioning. This ablation tests whether
    language conditioning in the bottleneck is necessary.

    Same interface as LanguageGuidedBottleneck (drop-in replacement).
    """

    def __init__(
        self,
        d_vlm: int = 1536,
        d_bottleneck: int = 512,
        n_queries: int = 16,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_bottleneck = d_bottleneck
        self.d_vlm = d_vlm

        self.queries = nn.Parameter(torch.randn(1, n_queries, d_bottleneck) * 0.02)
        self.tactile_proj = nn.Linear(512, d_bottleneck)
        # lang_proj kept for interface compat but unused in forward
        self.lang_proj = nn.Linear(d_vlm, d_bottleneck)

        self.layers = nn.ModuleList([
            PerceptionOnlyLayer(d_model=d_bottleneck, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_bottleneck),
            nn.Linear(d_bottleneck, d_vlm),
        )
        self.cls_proj = nn.Linear(d_vlm, d_bottleneck)

    def forward(
        self,
        lang_hidden: Tensor,  # [B, L, d_vlm] — ignored
        z_tactile: Tensor,    # [B, 512]
    ) -> Tensor:
        B = z_tactile.shape[0]

        tac_feats = self.tactile_proj(z_tactile).unsqueeze(1)
        perc_feats = tac_feats  # [B, 1, d_bn]

        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, None, perc_feats)

        return self.output_proj(queries)  # [B, K, d_vlm]

    def get_cls(self, bottleneck_tokens: Tensor) -> Tensor:
        return self.cls_proj(bottleneck_tokens[:, 0])
