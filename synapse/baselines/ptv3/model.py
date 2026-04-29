"""Baseline 3: Point Transformer V3 (Geometry Baseline).

Implements a point-attention mechanism inspired by Point Transformer V3
that operates on point cloud tokens.  Compares standard geometric
self-attention against the topological attention used in Deep Hodge.

Architecture:
    [B, N, d_model] → stack of PTv3Blocks
                    → global average pool → Linear → logits

Reference
---------
- Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", 2023
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from synapse.common.layers import DropPath, LayerScale, MLP

from ..base import BaselineBackbone
from ..registry import register_backbone


class PointAttention(nn.Module):
    """Point self-attention with distance-based positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pos_enc = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.attn_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        B, N, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, hd).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if coords is not None:
            rel = coords.unsqueeze(2) - coords.unsqueeze(1)
            pos_bias = self.pos_enc(rel)
            pos_bias = pos_bias.reshape(B, N, N, H, hd)
            pos_bias = pos_bias.permute(0, 3, 1, 2, 4)
            attn = attn + pos_bias.sum(dim=-1) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class PTv3Block(nn.Module):
    """Point Transformer V3 block with attention + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PointAttention(d_model, num_heads, dropout)
        self.ls1 = LayerScale(d_model, layer_scale_init)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_model * ffn_ratio, d_model, dropout)
        self.ls2 = LayerScale(d_model, layer_scale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor, coords: Tensor | None = None) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), coords)))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x


class PTv3Backbone(BaselineBackbone):
    """Point Transformer V3 baseline.

    Parameters
    ----------
    d_model : int
    num_classes : int
    num_layers : int
    num_heads : int
    ffn_ratio : int
    dropout : float
    drop_path : float
    use_coords : bool
    """

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        use_coords: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_coords = use_coords

        dp_rates = [drop_path * i / max(num_layers, 1) for i in range(num_layers)]

        self.blocks = nn.ModuleList([
            PTv3Block(d_model, num_heads, ffn_ratio, dropout, dp_rates[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @property
    def backbone_name(self) -> str:
        return "ptv3"

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        coords = x[:, :, :3] if self.use_coords else None

        for block in self.blocks:
            x = block(x, coords)

        x = self.norm(x)
        features = x.mean(dim=1)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits


register_backbone("ptv3", PTv3Backbone)


__all__ = ["PTv3Backbone"]
