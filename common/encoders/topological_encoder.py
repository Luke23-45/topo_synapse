"""Topological Encoder — Deep Hodge preprocessing pipeline.

Encapsulates the full Z3 topological preprocessing: event detection,
anchor selection, geometric lift, and topology projection.  Outputs
``[B, K, d_model]`` tokens ready for the ``DeepHodgeTransformer``
backbone.

This encoder is used *only* by the proposed Deep Hodge model.  All
other baselines use the simpler modality-specific encoders.

Input:  [B, T, input_dim] raw sequence
Output: [B, K, d_model]   topologically-encoded anchor tokens
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor

from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.lift import (
    dense_anchor_vectors,
    topology_project_torch,
)
from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.synapse_arch.normalized_lift import NormalizedLift


class TopologicalEncoder(nn.Module):
    """Z3 topological preprocessing encoder.

    Pipeline:
        1. CausalEventModel → event scores + saliency
        2. Relaxed selector → continuous anchor weights y*
        3. Dense anchor vectors → topology projection Π_top
        4. NormalizedLift → lifted cloud [B, T, k]
        5. Top-K anchor selection + projection → [B, K, d_model]

    Parameters
    ----------
    input_dim : int
        Dimensionality of each observation vector.
    d_model : int
        Output token dimension.
    hidden_dim : int
        Hidden dimension for the event model.
    k : int
        Latent geometric (lift) dimension.
    K : int
        Maximum number of retained anchors.
    r : int
        Refractory separation between retained anchors.
    lam : float
        Selector regularization weight.
    max_proxy_points : int
        Maximum simplicial complex size (K_eff).
    """

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

        anchor_dim = input_dim + 3  # [time, state, delta, saliency]

        # Stage 1: Event detection + saliency
        self.event_model = CausalEventModel(input_dim, hidden_dim)

        # Stage 2: Geometric lift
        self.lift = NormalizedLift(anchor_dim, k)

        # Stage 3: Project lifted k-dim to d_model
        self.topology_proj = nn.Linear(k, d_model)

    def set_normalization(self, mu: Tensor, sigma: Tensor) -> None:
        """Set lift normalization statistics (call before training)."""
        self.lift.set_normalization(mu, sigma)

    def _solve_batch_selector(self, saliency_scores: Tensor) -> Tensor:
        """Solve the relaxed selector QP for each sample in the batch."""
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode raw sequence into topological anchor tokens.

        Parameters
        ----------
        x : Tensor
            Raw sequence ``[B, T, input_dim]``.

        Returns
        -------
        tokens : Tensor
            Encoded anchor tokens ``[B, K_eff, d_model]``.
        y_star : Tensor
            Anchor weights ``[B, T]`` (needed for auxiliary losses).
        """
        # Stage 1: Event detection + saliency
        event_scores, saliency_scores = self.event_model(x)

        # Stage 2: Anchor selection
        y_star = self._solve_batch_selector(saliency_scores)

        # Stage 3: Topology-projected lift
        dense_vectors = dense_anchor_vectors(x, saliency_scores)
        dense_vectors = topology_project_torch(dense_vectors)
        _, dense_lifted_cloud = self.lift(dense_vectors)  # [B, T, k]

        # Stage 4: Select top-K anchors
        B, N, k_dim = dense_lifted_cloud.shape
        K_eff = min(N, self.max_proxy_points)
        _, top_idx = torch.topk(y_star, K_eff, dim=1)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim)
        cloud = torch.gather(dense_lifted_cloud, 1, top_idx_exp)  # [B, K_eff, k]

        # Stage 5: Project to d_model
        tokens = self.topology_proj(cloud)  # [B, K_eff, d_model]

        return tokens, y_star


__all__ = ["TopologicalEncoder"]
