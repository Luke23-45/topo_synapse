"""
Deep Hodge Transformer architecture.

This module implements a stack of Hodge-Laplacian routing layers that build
dynamic simplicial operators from the current token geometry and diffuse
messages across both nodes and edges.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepHodgeLayer(nn.Module):
    """Single Hodge routing layer."""

    def __init__(
        self,
        d_model: int,
        k_dim: int = 16,
        num_scales: int = 3,
        max_points: int = 16,
        tau: float = 1e-4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.k_dim = k_dim
        self.num_scales = num_scales
        self.max_points = max_points
        self.tau = tau

        self.geom_proj = nn.Linear(d_model, k_dim)
        self.log_scales = nn.Parameter(torch.linspace(-1.5, 1.5, num_scales))

        self.W_V0 = nn.Linear(d_model, d_model * num_scales)
        self.W_E_in = nn.Linear(d_model, d_model * num_scales)
        self.out_proj = nn.Linear(d_model * num_scales * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self._build_boundary_matrices(max_points)

        num_edges = max_points * (max_points - 1) // 2
        self.register_buffer("_eye_K", torch.eye(max_points))
        self.register_buffer("_eye_E", torch.eye(num_edges))

    def _build_boundary_matrices(self, K: int) -> None:
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                edges.append((i, j))

        triangles = []
        for i in range(K):
            for j in range(i + 1, K):
                for k in range(j + 1, K):
                    triangles.append((i, j, k))

        num_edges = len(edges)
        num_triangles = len(triangles)

        B1 = torch.zeros(K, num_edges)
        for e_idx, (i, j) in enumerate(edges):
            B1[i, e_idx] = -1.0
            B1[j, e_idx] = 1.0

        edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
        B2 = torch.zeros(num_edges, num_triangles)
        for t_idx, (i, j, k) in enumerate(triangles):
            B2[edge_to_idx[(j, k)], t_idx] = 1.0
            B2[edge_to_idx[(i, k)], t_idx] = -1.0
            B2[edge_to_idx[(i, j)], t_idx] = 1.0

        self.register_buffer("B1", B1)
        self.register_buffer("B2", B2)
        self.register_buffer("abs_B1", B1.abs())
        self.register_buffer("e_idx_i", torch.tensor([e[0] for e in edges], dtype=torch.long))
        self.register_buffer("e_idx_j", torch.tensor([e[1] for e in edges], dtype=torch.long))
        self.register_buffer(
            "t_idx_ij",
            torch.tensor([edge_to_idx[(t[0], t[1])] for t in triangles], dtype=torch.long),
        )
        self.register_buffer(
            "t_idx_jk",
            torch.tensor([edge_to_idx[(t[1], t[2])] for t in triangles], dtype=torch.long),
        )
        self.register_buffer(
            "t_idx_ik",
            torch.tensor([edge_to_idx[(t[0], t[2])] for t in triangles], dtype=torch.long),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_batch, K_eff, d = x.shape
        if K_eff > self.max_points:
            raise ValueError(f"Expected K<={self.max_points}, got {K_eff}")

        pad_len = self.max_points - K_eff
        if pad_len > 0:
            x_pad = torch.zeros(B_batch, pad_len, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, x_pad], dim=1)
            if mask is not None:
                m_pad = torch.zeros(B_batch, pad_len, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, m_pad], dim=1)

        if mask is None:
            mask = torch.ones(B_batch, self.max_points, device=x.device, dtype=x.dtype)
        else:
            mask = mask.to(device=x.device, dtype=x.dtype)

        residual = x
        x_norm = self.norm1(x)

        P = self.geom_proj(x_norm)
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        D_sq = torch.cdist(P, P).square() * mask_2d
        scales = torch.exp(self.log_scales)

        V0 = self.W_V0(x_norm).view(B_batch, self.max_points, self.num_scales, d).permute(0, 2, 1, 3)
        E_in_raw = torch.matmul(self.abs_B1.t(), x_norm)
        E_in = self.W_E_in(E_in_raw).view(B_batch, self.abs_B1.shape[1], self.num_scales, d).permute(0, 2, 1, 3)

        sigma_sq = scales.square().view(1, self.num_scales, 1, 1)
        affinity = torch.exp(-D_sq.unsqueeze(1) / (2.0 * sigma_sq + 1e-8)) * mask_2d.unsqueeze(1)

        W1 = affinity[:, :, self.e_idx_i, self.e_idx_j]
        W2 = W1[:, :, self.t_idx_ij] * W1[:, :, self.t_idx_jk] * W1[:, :, self.t_idx_ik]

        L0 = torch.einsum("ke,bse,le->bskl", self.B1, W1, self.B1)
        L0 = L0 + self.tau * self._eye_K.view(1, 1, self.max_points, self.max_points)
        node_updates = torch.einsum("bskl,bsld->bskd", L0, V0)

        term_down = torch.einsum("ek,bk,kf->bef", self.B1.t(), mask, self.B1)
        term_up = torch.einsum("et,bst,ft->bsef", self.B2, W2, self.B2)
        L1 = term_down.unsqueeze(1) + term_up
        L1 = L1 + self.tau * self._eye_E.view(1, 1, self._eye_E.shape[0], self._eye_E.shape[1])
        edge_messages = torch.einsum("bsef,bsfd->bsed", L1, E_in)
        edge_updates = torch.einsum("ke,bsed->bskd", self.abs_B1, edge_messages)

        node_updates = node_updates.permute(0, 2, 1, 3).reshape(B_batch, self.max_points, self.num_scales * d)
        edge_updates = edge_updates.permute(0, 2, 1, 3).reshape(B_batch, self.max_points, self.num_scales * d)
        attn_out = self.out_proj(torch.cat([node_updates, edge_updates], dim=-1))
        attn_out = self.dropout(attn_out)

        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))

        if pad_len > 0:
            x = x[:, :K_eff, :]

        return x


class DeepHodgeTransformer(nn.Module):
    """Stack of Hodge routing layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        k_dim: int = 16,
        num_scales: int = 3,
        max_points: int = 16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DeepHodgeLayer(d_model, k_dim, num_scales, max_points) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)
