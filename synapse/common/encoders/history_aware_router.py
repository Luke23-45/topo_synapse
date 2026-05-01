"""History-Aware Anchor Router (Z4).

Implements the learned anchor selection mechanism described in
``docs/formal_math/z4/encoder.md``.  The router replaces the hand-coded
saliency + QP pipeline with a fully differentiable, history-aware routing
block that learns which anchors to select and maintains a latent memory
state across routing stages.

Key components
--------------
- ``CandidateEncoder``     : embeds raw input into candidate tokens u_i
- ``AnchorScorer``         : computes anchor scores with positional + coverage bias
- ``DifferentiableSelector``: dense routing weights with diversity control
- ``SelectionStatistics``   : computes coverage, entropy, spacing, compactness
- ``HistoryAwareAnchorRouter``: full router with GRU memory update
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...synapse_core.topology_features import (
    append_selection_weight,
    build_feature_similarity,
    build_router_context,
    build_static_structural_features,
    build_structural_feature_tensor,
    precompute_structural_geometry,
    router_context_dim,
    structural_feature_dim,
)


# ---------------------------------------------------------------------------
# 1. Candidate encoding
# ---------------------------------------------------------------------------


class CandidateEncoder(nn.Module):
    """Embed the full input sequence into candidate tokens u_i.

    u_i = W_u x_i + b_u  ∈ ℝ^{d_u}
    """

    def __init__(self, input_dim: int, d_u: int, knn_k: int = 4) -> None:
        super().__init__()
        self.knn_k = knn_k
        feature_dim = structural_feature_dim(input_dim, include_selection=True)
        context_dim = router_context_dim(input_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim + context_dim),
            nn.Linear(feature_dim + context_dim, d_u),
            nn.GELU(),
            nn.Linear(d_u, d_u),
        )

    def project(self, structural_features: Tensor, context: Tensor, context_expanded: Optional[Tensor] = None) -> Tensor:
        if context_expanded is None:
            context_expanded = context.unsqueeze(1).expand(-1, structural_features.shape[1], -1)
        return self.proj(torch.cat([structural_features, context_expanded], dim=-1))

    def forward(
        self,
        x: Tensor,
        selection_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        geometry_cache: Optional[dict[str, Tensor]] = None,
        context: Optional[Tensor] = None,
        context_expanded: Optional[Tensor] = None,
        similarity: Optional[Tensor] = None,
        static_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor  [B, T, input_dim]
        static_features : Tensor | None  [B, T, d_static]
            Pre-built static features (without selection).  If provided,
            only the cheap ``append_selection_weight`` is called instead
            of the full ``build_structural_feature_tensor``.

        Returns
        -------
        u : Tensor  [B, T, d_u]
        structural_features : Tensor [B, T, d_f]
        similarity : Tensor [B, T, T]
        """
        if static_features is not None:
            structural_features = append_selection_weight(static_features, selection_weights)
        else:
            structural_features = build_structural_feature_tensor(
                x,
                selection_weights=selection_weights,
                mask=mask,
                knn_k=self.knn_k,
                geometry_cache=geometry_cache,
            )
        if context is None:
            context = build_router_context(x, mask=mask, geometry_cache=geometry_cache)
        u = self.project(structural_features, context, context_expanded=context_expanded)
        if similarity is None:
            if static_features is not None:
                similarity = build_feature_similarity(static_features, mask=mask)
            else:
                sf = build_structural_feature_tensor(
                    x,
                    mask=mask,
                    knn_k=self.knn_k,
                    geometry_cache=geometry_cache,
                    include_selection=False,
                )
                similarity = build_feature_similarity(sf, mask=mask)
        return u, structural_features, similarity, context


# ---------------------------------------------------------------------------
# 2. Anchor scoring
# ---------------------------------------------------------------------------


class AnchorScorer(nn.Module):
    """Compute anchor scores  a_i^(ℓ) = <q, k_i>/√d_a + b_pos(i) + b_cov(i).

    Parameters
    ----------
    d_u : int
        Candidate token dimension.
    d_m : int
        Memory state dimension (for query projection).
    d_a : int
        Attention / scoring dimension.
    coverage_gamma : float
        Strength of coverage bias (γ ≥ 0).
    """

    def __init__(
        self,
        d_u: int,
        d_m: int,
        d_a: int,
        max_seq_len: int = 512,
        coverage_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_a = d_a
        self.coverage_gamma = coverage_gamma

        self.W_q = nn.Linear(d_m, d_a, bias=False)
        self.W_k = nn.Linear(d_u, d_a, bias=False)
        self.W_v = nn.Linear(d_u, d_u, bias=False)
        self.structure_bias = nn.Sequential(
            nn.Linear(3, d_a),
            nn.GELU(),
            nn.Linear(d_a, 1),
        )

    def forward(
        self,
        u: Tensor,
        structural_features: Tensor,
        similarity: Tensor,
        memory: Tensor,
        prev_selected: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute anchor scores and value vectors.

        Parameters
        ----------
        u : Tensor  [B, T, d_u]
            Candidate tokens.
        memory : Tensor  [B, d_m]
            Current router memory state m^(ℓ).
        prev_selected : Tensor | None  [B, T]
            Cumulative selection mask from previous stages (for coverage bias).

        Returns
        -------
        scores : Tensor  [B, T]
            Anchor scores a_i^(ℓ).
        v : Tensor  [B, T, d_u]
            Value vectors v_i = W_v u_i.
        """
        B, T, _ = u.shape

        # q^(ℓ) = W_q m^(ℓ)   →  [B, d_a]
        q = self.W_q(memory)  # [B, d_a]

        # k_i = W_k u_i   →  [B, T, d_a]
        k = self.W_k(u)  # [B, T, d_a]

        # v_i = W_v u_i   →  [B, T, d_u]
        v = self.W_v(u)  # [B, T, d_u]

        # Scaled dot-product: <q, k_i> / √d_a  →  [B, T]
        scale = math.sqrt(self.d_a)
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / scale  # [B, T]

        descriptors = structural_features[..., -4:-1]
        scores = scores + self.structure_bias(descriptors).squeeze(-1)

        # Coverage bias from feature-space overlap with previously selected anchors.
        if prev_selected is not None and self.coverage_gamma > 0:
            coverage_map = torch.bmm(similarity, prev_selected.unsqueeze(-1)).squeeze(-1)
            scores = scores - self.coverage_gamma * coverage_map

        return scores, v


# ---------------------------------------------------------------------------
# 3. Differentiable selection
# ---------------------------------------------------------------------------


class DifferentiableSelector(nn.Module):
    """Dense routing weights with feature-space diversity control.

    Produces a normalized distribution y over all input positions for each
    routing stage. This is not anchor selection: every valid input can
    contribute to the stage token through its routing mass.
    """

    def __init__(
        self,
        K: int,
        r: int = 1,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature))
        )

    def forward(
        self,
        scores: Tensor,
        similarity: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        hard: bool = False,
    ) -> Tensor:
        """Compute selection weights.

        Parameters
        ----------
        scores : Tensor  [B, T]
            Anchor scores a^(ℓ).
        hard : bool
            Accepted for API compatibility. The Z4 path keeps routing soft.

        Returns
        -------
        y : Tensor  [B, T]
            Dense routing weights summing to 1 over valid positions.
        """
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)

        # Dense softmax routing with learned temperature
        logits = scores / temp
        if mask is not None:
            invalid = mask <= 0
            logits = logits.masked_fill(invalid, float("-inf"))

        y = torch.softmax(logits, dim=1)

        # Diversity constraint in feature space rather than index space.
        if similarity is not None and self.r > 0:
            for _ in range(self.r):
                overlap = torch.bmm(similarity, y.unsqueeze(-1)).squeeze(-1)
                penalty = overlap / overlap.amax(dim=1, keepdim=True).clamp_min(1e-6)
                logits = logits - penalty
                if mask is not None:
                    logits = logits.masked_fill(mask <= 0, float("-inf"))
                y = torch.softmax(logits, dim=1)

        return y

# ---------------------------------------------------------------------------
# 4. Selection-history statistics
# ---------------------------------------------------------------------------


class SelectionStatistics(nn.Module):
    """Compute selection-history statistics c^(ℓ) = Ψ(y, u).

    Statistics:
    - coverage   : (1/T) ||y||_1
    - entropy    : H(ȳ) where ȳ = y / (||y||_1 + ε)
    - spacing    : Spread(y) — average gap between selected positions
    - compactness: Compactness(u, y) — weighted average pairwise distance

    Output dimension: d_c = 4 (one scalar per statistic).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y: Tensor, u: Tensor) -> Tensor:
        """Compute selection statistics.

        Parameters
        ----------
        y : Tensor  [B, T]
            Selection weights.
        u : Tensor  [B, T, d_u]
            Candidate tokens.

        Returns
        -------
        c : Tensor  [B, 4]
            Concatenated statistics [coverage, entropy, spacing, compactness].
        """
        B, T, d_u = u.shape

        # Coverage: (1/T) ||y||_1
        coverage = y.sum(dim=1) / T  # [B]

        # Entropy: H(ȳ) where ȳ = y / (||y||_1 + ε)
        y_sum = y.sum(dim=1, keepdim=True).clamp(min=self.eps)
        y_norm = y / y_sum  # [B, T]
        y_norm_clamped = y_norm.clamp(min=self.eps)  # avoid log(0)
        entropy = -(y_norm * y_norm_clamped.log()).sum(dim=1)  # [B]

        # Spread: weighted distance to selected centroid in token space
        spacing = self._compute_spread(y, u)  # [B]

        # Compactness: weighted average pairwise distance
        compactness = self._compute_compactness(y, u)  # [B]

        return torch.stack([coverage, entropy, spacing, compactness], dim=1)  # [B, 4]

    def _compute_spread(self, y: Tensor, u: Tensor) -> Tensor:
        """Compute weighted spread around the selected centroid."""
        total_weight = y.sum(dim=1, keepdim=True).clamp(min=self.eps)
        centroid = torch.bmm(y.unsqueeze(1), u).squeeze(1) / total_weight
        dist = (u - centroid.unsqueeze(1)).norm(dim=-1)
        spread = (y * dist).sum(dim=1) / total_weight.squeeze(1)
        return spread

    def _compute_compactness(self, y: Tensor, u: Tensor) -> Tensor:
        """Compute weighted average pairwise squared distance (compactness).

        Uses the parallel-axis identity to avoid O(T²d_u) pairwise
        materialisation:

            Σ_{i,j} y_i y_j ||u_i - u_j||²
                = 2 (Σy · Σ_i y_i ||u_i||² − ||Σ_i y_i u_i||²)

        Output is divided by (Σ y_i)² so the statistic is scale-invariant
        w.r.t. the selection weights.  Squared distances are used instead
        of L2 for numerical stability and because the statistic only feeds
        a GRU — the exact norm flavour is not critical.
        """
        total_weight = y.sum(dim=1).clamp(min=self.eps)  # [B]

        # Weighted sum of squared norms: Σ_i y_i ||u_i||²   →  [B]
        weighted_sq_norm = (y * u.pow(2).sum(dim=-1)).sum(dim=1)

        # Weighted centroid norm: ||Σ_i y_i u_i||²   →  [B]
        centroid = torch.bmm(y.unsqueeze(1), u).squeeze(1)  # [B, d_u]
        centroid_sq_norm = centroid.pow(2).sum(dim=1)  # [B]

        # Parallel-axis identity
        weighted_dist_sq = 2.0 * (total_weight * weighted_sq_norm - centroid_sq_norm)

        compactness = weighted_dist_sq / total_weight.pow(2).clamp(min=self.eps)  # [B]
        return compactness


# ---------------------------------------------------------------------------
# 5. History-Aware Anchor Router
# ---------------------------------------------------------------------------


class HistoryAwareAnchorRouter(nn.Module):
    """History-Aware Anchor Router with GRU memory.

    Implements the full routing pipeline from ``docs/formal_math/z4/encoder.md``:

    1. Candidate encoding: u_i = E(x_i)
    2. Anchor scoring: a_i = <q, k_i>/√d_a + b_pos(i) + b_cov(i)
    3. Differentiable selection: y = S_{K,r,τ}(a)
    4. Content summary: z = Σ y_i v_i
    5. Statistics: c = Ψ(y, u)
    6. Memory update: m' = GRU(m, [z; c; r])

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input observation.
    d_u : int
        Candidate token dimension.
    d_a : int
        Anchor scoring dimension.
    d_m : int
        Memory state dimension.
    K : int
        Number of anchors to select per routing stage.
    r : int
        Refractory separation radius.
    L : int
        Number of routing stages.
    coverage_gamma : float
        Coverage bias strength.
    init_temperature : float
        Initial selection temperature.
    feedback_dim : int
        Dimension of the feedback signal r^(ℓ).
        max_seq_len : int
        Kept for API compatibility; unused by the structure-aware router.
    """

    def __init__(
        self,
        input_dim: int,
        d_u: int = 64,
        d_a: int = 32,
        d_m: int = 64,
        K: int = 8,
        r: int = 1,
        L: int = 16,
        coverage_gamma: float = 1.0,
        init_temperature: float = 1.0,
        feedback_dim: int = 1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_u = d_u
        self.d_m = d_m
        self.K = K
        self.r = r
        self.L = L

        # Sub-modules
        self.candidate_encoder = CandidateEncoder(input_dim, d_u, knn_k=max(1, r))
        self.scorer = AnchorScorer(d_u, d_m, d_a, max_seq_len, coverage_gamma)
        self.selector = DifferentiableSelector(K, r, init_temperature)
        self.statistics = SelectionStatistics()
        self.memory_init = nn.Sequential(
            nn.LayerNorm(router_context_dim(input_dim)),
            nn.Linear(router_context_dim(input_dim), d_m),
            nn.Tanh(),
        )

        # GRU for memory update
        # Input: [z; c; r] where z ∈ ℝ^{d_u}, c ∈ ℝ^4, r ∈ ℝ^{feedback_dim}
        self.feedback_dim = feedback_dim
        gru_input_dim = d_u + 4 + feedback_dim
        self.gru = nn.GRUCell(gru_input_dim, d_m)

    def forward(
        self,
        x: Tensor,
        feedback: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        hard: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Run the full routing pipeline across L stages.

        Parameters
        ----------
        x : Tensor  [B, T, input_dim]
            Raw input sequence.
        feedback : Tensor | None  [B, n_feedback]
            Training feedback signal r^(ℓ).  If None, zeros are used
            (inference mode).
        hard : bool
            Accepted for API compatibility. Routing remains soft.

        Returns
        -------
        all_y : Tensor  [B, L, T]
            Selection weights for each routing stage.
        all_z : Tensor  [B, L, d_u]
            Content summaries for each routing stage.
        all_memory : Tensor  [B, L+1, d_m]
            Memory states (including initial m^(0)).
        all_anchors : Tensor  [B, L, T, d_u]
            Selected anchor tokens (weighted by y) for each stage.
        geometry_cache : dict[str, Tensor]
            Pre-computed geometry (reusable by the encoder for the lift).
        """
        B, T, _ = x.shape
        device = x.device

        geometry_cache = precompute_structural_geometry(x, mask=mask, knn_k=max(1, self.r))
        context = build_router_context(x, mask=mask, geometry_cache=geometry_cache)
        static_features = build_static_structural_features(
            x,
            mask=mask,
            knn_k=max(1, self.r),
            geometry_cache=geometry_cache,
        )
        similarity = build_feature_similarity(static_features, mask=mask)

        # Initialize memory from global structure context.
        memory = self.memory_init(context).to(dtype=x.dtype)

        # Cumulative selection mask for coverage bias
        cumulative_y = torch.zeros(B, T, device=device, dtype=x.dtype)

        # Pre-expand context for candidate encoder (avoids L repeated unsqueeze+expand)
        context_expanded = context.unsqueeze(1).expand(-1, T, -1)  # [B, T, d_context]

        # Pre-allocate zero feedback tensor (avoids L allocations when feedback is None)
        if feedback is None:
            zero_feedback = torch.zeros(B, self.feedback_dim, device=device, dtype=x.dtype)

        # Storage for outputs
        all_y = []
        all_z = []
        all_memory = [memory]
        all_anchors = []

        for ell in range(self.L):
            # Stage 2: Anchor scoring
            stage_u, stage_features, stage_similarity, _ = self.candidate_encoder(
                x,
                selection_weights=cumulative_y,
                mask=mask,
                geometry_cache=geometry_cache,
                context=context,
                context_expanded=context_expanded,
                similarity=similarity,
                static_features=static_features,
            )
            scores, v = self.scorer(
                stage_u,
                stage_features,
                stage_similarity,
                memory,
                prev_selected=cumulative_y,
            )

            # Stage 3: Differentiable selection
            y = self.selector(scores, similarity=stage_similarity, mask=mask, hard=hard)  # [B, T]

            # Stage 4: Content summary
            z = torch.bmm(y.unsqueeze(1), v).squeeze(1)  # [B, d_u]

            # Stage 5: Selection-history statistics
            c = self.statistics(y, stage_u)  # [B, 4]

            # Stage 6: Memory update
            r_ell = feedback if feedback is not None else zero_feedback

            gru_input = torch.cat([z, c, r_ell], dim=-1)  # [B, d_u + 4 + n_feedback]
            memory = self.gru(gru_input, memory)  # [B, d_m]

            # Update cumulative selection for coverage bias
            cumulative_y = cumulative_y + y

            # Store outputs
            all_y.append(y)
            all_z.append(z)
            all_memory.append(memory)
            # Anchor tokens weighted by selection
            anchors = stage_u * y.unsqueeze(-1)  # [B, T, d_u]
            all_anchors.append(anchors)

        all_y = torch.stack(all_y, dim=1)  # [B, L, T]
        all_z = torch.stack(all_z, dim=1)  # [B, L, d_u]
        all_memory = torch.stack(all_memory, dim=1)  # [B, L+1, d_m]
        all_anchors = torch.stack(all_anchors, dim=1)  # [B, L, T, d_u]

        return all_y, all_z, all_memory, all_anchors, geometry_cache
