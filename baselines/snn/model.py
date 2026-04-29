"""Baseline 4: Simplicial Neural Network (Topology Baseline).

Implements a simplicial message passing network that operates on
0-simplices (nodes) and 1-simplices (edges) with boundary-operator-
based message passing.  Compares discrete simplicial convolutions
against the continuous Hodge-spectral approach in Deep Hodge.

Architecture:
    [B, N, d_model] → build KNN graph → stack of SNNBlocks
                    → global average pool → Linear → logits

Reference
---------
- Ebli et al., "Simplicial Neural Networks", 2020
- Hajij et al., "Simplicial Complex Attention Network", 2022
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from synapse.common.layers import DropPath, LayerScale, MLP

from ..base import BaselineBackbone
from ..registry import register_backbone


def knn_graph(coords: Tensor, k: int) -> Tensor:
    """Build KNN adjacency from coordinates.

    Returns
    -------
    edge_index : Tensor
        ``[2, B, N*k]`` — source and target indices per batch.
    """
    B, N, _ = coords.shape
    k = min(k, N - 1)

    dist = torch.cdist(coords, coords)
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    idx = idx[:, :, 1:]  # remove self, [B, N, k]

    src_base = torch.arange(N, device=coords.device).unsqueeze(1).expand_as(idx[0])
    src = src_base.unsqueeze(0).expand(B, -1, -1)

    edge_index = torch.stack([src.reshape(B, -1), idx.reshape(B, -1)], dim=0)
    return edge_index


class SimplicialConv(nn.Module):
    """Simplicial message passing over 0- and 1-simplices."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.node_msg = nn.Linear(d_model, d_model)
        self.edge_msg = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_to_node = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        B, N, D = x.shape
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        # Node-to-node (0-simplex)
        node_messages = self.node_msg(x)
        node_agg = torch.zeros_like(x)
        src_exp = src_idx.unsqueeze(-1).expand(-1, -1, D)
        tgt_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, D)
        gathered = torch.gather(node_messages, 1, src_exp)
        node_agg.scatter_add_(1, tgt_exp, gathered)
        node_agg = self.drop(node_agg)

        # Edge (1-simplex)
        src_feats = torch.gather(x, 1, src_exp)
        tgt_feats = torch.gather(x, 1, tgt_exp)
        edge_feats = torch.cat([src_feats, tgt_feats], dim=-1)
        edge_messages = self.edge_msg(edge_feats)
        edge_messages = self.drop(edge_messages)

        edge_agg = torch.zeros_like(x)
        edge_contrib = self.edge_to_node(edge_messages)
        edge_agg.scatter_add_(1, tgt_exp, edge_contrib)

        combined = torch.cat([node_agg, edge_agg], dim=-1)
        return self.out_proj(combined)


class SNNBlock(nn.Module):
    """Simplicial Neural Network residual block."""

    def __init__(
        self,
        d_model: int,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.simp_conv = SimplicialConv(d_model, dropout)
        self.ls1 = LayerScale(d_model, layer_scale_init)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_model * ffn_ratio, d_model, dropout)
        self.ls2 = LayerScale(d_model, layer_scale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.simp_conv(self.norm1(x), edge_index)))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x


class SNNBackbone(BaselineBackbone):
    """Simplicial Neural Network baseline.

    Parameters
    ----------
    d_model : int
    num_classes : int
    num_layers : int
    ffn_ratio : int
    dropout : float
    drop_path : float
    k_neighbors : int
    use_coords : bool
    """

    def __init__(
        self,
        d_model: int = 64,
        num_classes: int = 4,
        num_layers: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        k_neighbors: int = 16,
        use_coords: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_coords = use_coords
        self.k_neighbors = k_neighbors

        dp_rates = [drop_path * i / max(num_layers, 1) for i in range(num_layers)]

        self.blocks = nn.ModuleList([
            SNNBlock(d_model, ffn_ratio, dropout, dp_rates[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @property
    def backbone_name(self) -> str:
        return "snn"

    def forward(
        self,
        x: Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        coords = x[:, :, :3] if self.use_coords else x
        edge_index = knn_graph(coords, self.k_neighbors)

        for block in self.blocks:
            x = block(x, edge_index)

        x = self.norm(x)
        features = x.mean(dim=1)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits


register_backbone("snn", SNNBackbone)


__all__ = ["SNNBackbone"]
