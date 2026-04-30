"""Legacy Z3 topological encoder preserved under the legacy namespace."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor

from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.topology_features import (
    build_feature_similarity,
    build_structural_feature_tensor,
    structural_feature_dim,
)
from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.synapse_arch.normalized_lift import NormalizedLift


class SoftSelectorProxy(nn.Module):
    """Legacy differentiable GPU proxy for Z3 anchor selection."""

    def __init__(self, K: int, r: int, lam: float, init_temperature: float = 1.0):
        super().__init__()
        self.K = K
        self.r = r
        self.lam = lam
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    def forward(
        self,
        saliency: Tensor,
        similarity: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        logits = (saliency / (2.0 * self.lam) - 0.5) / temp
        y = torch.sigmoid(logits)

        if mask is not None:
            y = y * mask.to(dtype=y.dtype)

        budget = y.sum(dim=1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.K / budget, max=1.0)
        y = y * scale

        if similarity is not None and self.r > 0:
            for _ in range(self.r):
                overlap = torch.bmm(similarity, y.unsqueeze(-1)).squeeze(-1)
                y = y / (1.0 + overlap)
                budget = y.sum(dim=1, keepdim=True).clamp(min=1e-6)
                y = y * torch.clamp(self.K / budget, max=1.0)
        return y


class TopologicalEncoder(nn.Module):
    """Legacy Z3 topological preprocessing encoder."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: int = 64,
        k: int = 16,
        K: int = 8,
        r: int = 1,
        lam: float = 0.5,
        max_proxy_points: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.lam = lam
        self.max_proxy_points = max_proxy_points

        anchor_dim = structural_feature_dim(input_dim, include_selection=False)
        self.event_model = CausalEventModel(input_dim, hidden_dim)
        self.selector_proxy = SoftSelectorProxy(K=K, r=r, lam=lam)
        self.lift = NormalizedLift(anchor_dim, k)
        self.topology_proj = nn.Linear(k, d_model)

    def set_normalization(self, mu: Tensor, sigma: Tensor) -> None:
        self.lift.set_normalization(mu, sigma)

    def _solve_batch_selector(self, saliency_scores: Tensor) -> Tensor:
        y_values = []
        for scores in saliency_scores.detach().cpu().numpy():
            y_values.append(
                solve_relaxed_selector(
                    scores.astype(np.float64),
                    K=self.K,
                    r=self.r,
                    lam=self.lam,
                )
            )
        return torch.from_numpy(np.stack(y_values, axis=0)).to(
            device=saliency_scores.device,
            dtype=saliency_scores.dtype,
        )

    def _gpu_selector(
        self,
        saliency_scores: Tensor,
        similarity: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.selector_proxy(saliency_scores, similarity=similarity, mask=mask)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        _, saliency_scores = self.event_model(x)

        structural_features = build_structural_feature_tensor(
            x,
            selection_weights=saliency_scores,
            knn_k=max(1, self.r),
        )
        similarity = build_feature_similarity(structural_features)
        y_star = self._gpu_selector(saliency_scores, similarity=similarity)

        dense_vectors = build_structural_feature_tensor(
            x,
            knn_k=max(1, self.r),
            include_selection=False,
        )
        _, dense_lifted_cloud = self.lift(dense_vectors)

        _, N, k_dim = dense_lifted_cloud.shape
        K_eff = min(N, self.max_proxy_points)
        _, top_idx = torch.topk(y_star, K_eff, dim=1)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
        cloud = torch.gather(dense_lifted_cloud, 1, top_idx_exp)
        tokens = self.topology_proj(cloud)
        return tokens, y_star


__all__ = ["TopologicalEncoder"]
