"""Z4 Topological Encoder with History-Aware Anchor Router.

Replaces the Z3 ``TopologicalEncoder`` (saliency + QP) with a fully
learned, history-aware routing pipeline.  The router selects anchors
adaptively and feeds them into the geometric lift and Hodge branch.

Pipeline
--------
1. HistoryAwareAnchorRouter → selection weights y, content summaries z,
   memory states m, anchor tokens
2. Dense structure-aware vectors
3. NormalizedLift → lifted cloud [B, T, k]
4. Stage-wise weighted pooling + projection → [B, L, d_model]

The router's memory update is fed by task and topology feedback so that
anchor choice evolves together with representation learning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .history_aware_router import HistoryAwareAnchorRouter
from ...synapse_core.topology_features import (
    build_static_structural_features,
    build_structural_feature_tensor,
    structural_feature_dim,
)
from ...synapse_arch.normalized_lift import NormalizedLift


class Z4TopologicalEncoder(nn.Module):
    """Z4 topological preprocessing encoder with history-aware routing.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each observation vector.
    d_model : int
        Output token dimension.
    d_u : int
        Candidate token dimension (router internal).
    d_a : int
        Anchor scoring dimension (router internal).
    d_m : int
        Memory state dimension (router internal).
    hidden_dim : int
        Hidden dimension (kept for API compatibility; unused if event_model is None).
    k : int
        Latent geometric (lift) dimension.
    K : int
        Number of anchors to select per routing stage.
    r : int
        Refractory separation between retained anchors.
    L : int
        Number of routing stages.
    lam : float
        Selector regularization weight (kept for API compatibility).
    coverage_gamma : float
        Coverage bias strength.
    init_temperature : float
        Initial selection temperature.
    feedback_dim : int
        Dimension of the feedback signal for the router's GRU.
    max_proxy_points : int
        Kept for API compatibility; unused by the soft Z4 encoder path.
    max_seq_len : int
        Maximum sequence length for positional bias.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: int = 64,
        d_u: int = 64,
        d_a: int = 32,
        d_m: int = 64,
        k: int = 16,
        K: int = 8,
        r: int = 1,
        L: int = 16,
        lam: float = 0.5,
        coverage_gamma: float = 1.0,
        init_temperature: float = 1.0,
        feedback_dim: int = 1,
        max_proxy_points: int = 16,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.L = L
        anchor_dim = structural_feature_dim(input_dim, include_selection=False)

        # Stage 1: History-Aware Anchor Router
        self.router = HistoryAwareAnchorRouter(
            input_dim=input_dim,
            d_u=d_u,
            d_a=d_a,
            d_m=d_m,
            K=K,
            r=r,
            L=L,
            coverage_gamma=coverage_gamma,
            init_temperature=init_temperature,
            feedback_dim=feedback_dim,
            max_seq_len=max_seq_len,
        )

        # Stage 2: Geometric lift (reuses existing modules)
        self.lift = NormalizedLift(anchor_dim, k)

        # Stage 3: Project lifted k-dim to d_model
        self.topology_proj = nn.Linear(k, d_model)

    def set_normalization(self, mu: Tensor, sigma: Tensor) -> None:
        """Set lift normalization statistics (call before training)."""
        self.lift.set_normalization(mu, sigma)

    def forward(
        self,
        x: Tensor,
        feedback: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode raw sequence into topological anchor tokens.

        Parameters
        ----------
        x : Tensor  [B, T, input_dim]
            Raw input sequence.
        feedback : Tensor | None  [B, n_feedback]
            Training feedback signal for the router's memory update.
            Typically [λ_task * L_task + λ_topo * L_topo + λ_reg * L_reg].
            If None, the router uses zero feedback (inference mode).

        Returns
        -------
        tokens : Tensor  [B, L, d_model]
            Encoded anchor tokens.
        y_star : Tensor  [B, T]
            Mean routing weights aggregated across stages.
        all_y : Tensor  [B, L, T]
            Per-stage selection weights.
        all_memory : Tensor  [B, L+1, d_m]
            Memory states across routing stages.
        """
        # Stage 1: Router — learned anchor selection
        all_y, all_z, all_memory, all_anchors, geometry_cache = self.router(
            x, feedback=feedback, mask=mask, hard=False
        )

        # Aggregate routing weights across stages without changing total mass.
        y_star = all_y.mean(dim=1)  # [B, T]

        # Stage 2: Build structure-aware dense vectors for the lift
        # Reuse geometry_cache from the router to avoid redundant computation.
        dense_vectors = build_static_structural_features(
            x,
            mask=mask,
            knn_k=max(1, self.r),
            geometry_cache=geometry_cache,
        )
        _, dense_lifted_cloud = self.lift(dense_vectors)  # [B, T, k]

        # Stage 3: Convert each routing stage into a soft anchor token.
        stage_mass = all_y.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, L, 1]
        cloud = torch.einsum("blt,btk->blk", all_y, dense_lifted_cloud) / stage_mass  # [B, L, k]

        # Stage 4: Project to d_model
        tokens = self.topology_proj(cloud)  # [B, L, d_model]

        return tokens, y_star, all_y, all_memory
