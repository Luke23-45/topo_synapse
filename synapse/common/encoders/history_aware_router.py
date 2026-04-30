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
- ``DifferentiableSelector``: relaxed top-K selection with refractory radius
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


# ---------------------------------------------------------------------------
# 1. Candidate encoding
# ---------------------------------------------------------------------------


class CandidateEncoder(nn.Module):
    """Embed the full input sequence into candidate tokens u_i.

    u_i = W_u x_i + b_u  ∈ ℝ^{d_u}
    """

    def __init__(self, input_dim: int, d_u: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_u)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, T, input_dim]

        Returns
        -------
        u : Tensor  [B, T, d_u]
        """
        return self.proj(x)


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
    max_seq_len : int
        Maximum sequence length (for positional bias table).
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

        # Learnable positional bias: one scalar per position
        self.b_pos = nn.Parameter(torch.zeros(max_seq_len))

    def forward(
        self,
        u: Tensor,
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

        # Positional bias: b_pos(i) for i ∈ [0, T)
        pos_bias = self.b_pos[:T].unsqueeze(0).expand(B, -1)  # [B, T]
        scores = scores + pos_bias

        # Coverage bias: b_cov^(ℓ)(i) = -γ · d(i, I^(<ℓ))
        if prev_selected is not None and self.coverage_gamma > 0:
            # Approximate coverage distance: if position i was already
            # selected (prev_selected > 0), penalize it.
            # Distance proxy: 1 - min distance to selected set.
            # Simple form: penalize positions that are close to already-selected ones.
            # We use a convolution-like smoothing of prev_selected to get
            # a soft "already-covered" map, then negate with gamma.
            coverage_map = prev_selected  # [B, T], values in [0, 1]
            scores = scores - self.coverage_gamma * coverage_map

        return scores, v


# ---------------------------------------------------------------------------
# 3. Differentiable selection
# ---------------------------------------------------------------------------


class DifferentiableSelector(nn.Module):
    """Relaxed top-K selection with refractory radius.

    S_{K,r,τ}(a) produces continuous weights y ∈ [0,1]^T that:
    - approximately select K positions,
    - enforce a refractory separation of at least r between selected positions,
    - are differentiable w.r.t. the scores a.

    At inference, hard selection is used instead.
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

    def forward(self, scores: Tensor, hard: bool = False) -> Tensor:
        """Compute selection weights.

        Parameters
        ----------
        scores : Tensor  [B, T]
            Anchor scores a^(ℓ).
        hard : bool
            If True, use hard (non-differentiable) selection for inference.

        Returns
        -------
        y : Tensor  [B, T]
            Selection weights in [0, 1].
        """
        B, T = scores.shape
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)

        if hard:
            return self._hard_select(scores)

        # Soft relaxed selection via sigmoid with temperature
        logits = scores / temp
        y = torch.sigmoid(logits)

        # Soft budget constraint: scale down if sum exceeds K
        budget = y.sum(dim=1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.K / budget, max=1.0)
        y = y * scale

        # Soft refractory constraint: dampen overlapping selections within radius r
        if self.r > 0:
            for d in range(1, min(self.r + 1, T)):
                shift = torch.roll(y, shifts=-d, dims=1)
                pair_sum = y + shift
                damping = torch.clamp(2.0 / (1.0 + pair_sum), max=1.0)
                y = y * damping

        # First position constraint (causal: cannot select t=0)
        y = y.clone()
        y[:, 0] = 0.0

        return y

    def _hard_select(self, scores: Tensor) -> Tensor:
        """Hard top-K with refractory radius (inference only).

        Greedily selects the top-scoring position, then masks out
        its neighbours within radius r, and repeats.
        """
        B, T = scores.shape
        y = torch.zeros_like(scores)
        scores_masked = scores.clone()
        scores_masked[:, 0] = float("-inf")

        for _ in range(self.K):
            # Pick highest-scoring position
            top_idx = scores_masked.argmax(dim=1)  # [B]
            arange = torch.arange(B, device=scores.device)
            y[arange, top_idx] = 1.0

            # Mask out neighbours within radius r
            for d in range(-self.r, self.r + 1):
                if d == 0:
                    continue
                neighbour_idx = top_idx + d
                valid = (neighbour_idx >= 0) & (neighbour_idx < T)
                safe_idx = neighbour_idx.clamp(0, T - 1)
                scores_masked[arange, safe_idx] = torch.where(
                    valid, float("-inf"), scores_masked[arange, safe_idx]
                )

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

        # Spacing: average gap between top-selected positions
        spacing = self._compute_spacing(y)  # [B]

        # Compactness: weighted average pairwise distance
        compactness = self._compute_compactness(y, u)  # [B]

        return torch.stack([coverage, entropy, spacing, compactness], dim=1)  # [B, 4]

    def _compute_spacing(self, y: Tensor) -> Tensor:
        """Compute average gap between selected positions.

        Uses soft selection: positions with y > threshold contribute.
        """
        B, T = y.shape
        # Get soft positions
        weights = y  # [B, T]
        positions = torch.arange(T, device=y.device, dtype=y.dtype).unsqueeze(0).expand(B, -1)  # [B, T]

        # Weighted mean position
        total_weight = weights.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [B, 1]
        mean_pos = (weights * positions).sum(dim=1, keepdim=True) / total_weight  # [B, 1]

        # Weighted variance of positions (proxy for spread)
        variance = (weights * (positions - mean_pos) ** 2).sum(dim=1) / total_weight.squeeze(1)  # [B]
        spacing = variance.clamp(min=self.eps).sqrt()  # [B]

        return spacing

    def _compute_compactness(self, y: Tensor, u: Tensor) -> Tensor:
        """Compute weighted average pairwise distance (compactness).

        Compactness(u, y) = Σ_{i,j} y_i y_j ||u_i - u_j|| / (Σ y_i)^2
        """
        B, T, d_u = u.shape

        # Pairwise distances: ||u_i - u_j||
        # u: [B, T, d_u] → diff: [B, T, T, d_u]
        diff = u.unsqueeze(2) - u.unsqueeze(1)  # [B, T, T, d_u]
        dist = ((diff ** 2).sum(dim=-1) + self.eps).sqrt()  # [B, T, T]

        # Weight by y_i * y_j
        y_outer = y.unsqueeze(1) * y.unsqueeze(2)  # [B, T, T]
        weighted_dist = (y_outer * dist).sum(dim=(1, 2))  # [B]

        total_weight = y.sum(dim=1).clamp(min=self.eps)  # [B]
        compactness = weighted_dist / total_weight.pow(2).clamp(min=self.eps)  # [B]

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
        Maximum sequence length for positional bias.
    """

    def __init__(
        self,
        input_dim: int,
        d_u: int = 64,
        d_a: int = 32,
        d_m: int = 64,
        K: int = 8,
        r: int = 1,
        L: int = 1,
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
        self.candidate_encoder = CandidateEncoder(input_dim, d_u)
        self.scorer = AnchorScorer(d_u, d_m, d_a, max_seq_len, coverage_gamma)
        self.selector = DifferentiableSelector(K, r, init_temperature)
        self.statistics = SelectionStatistics()

        # GRU for memory update
        # Input: [z; c; r] where z ∈ ℝ^{d_u}, c ∈ ℝ^4, r ∈ ℝ^{feedback_dim}
        self.feedback_dim = feedback_dim
        gru_input_dim = d_u + 4 + feedback_dim
        self.gru = nn.GRUCell(gru_input_dim, d_m)

    def forward(
        self,
        x: Tensor,
        feedback: Optional[Tensor] = None,
        hard: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the full routing pipeline across L stages.

        Parameters
        ----------
        x : Tensor  [B, T, input_dim]
            Raw input sequence.
        feedback : Tensor | None  [B, n_feedback]
            Training feedback signal r^(ℓ).  If None, zeros are used
            (inference mode).
        hard : bool
            If True, use hard selection (inference).

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
        """
        B, T, _ = x.shape
        device = x.device

        # Stage 1: Candidate encoding
        u = self.candidate_encoder(x)  # [B, T, d_u]

        # Initialize memory: m^(0) = 0
        memory = torch.zeros(B, self.d_m, device=device, dtype=x.dtype)

        # Cumulative selection mask for coverage bias
        cumulative_y = torch.zeros(B, T, device=device, dtype=x.dtype)

        # Storage for outputs
        all_y = []
        all_z = []
        all_memory = [memory]
        all_anchors = []

        for ell in range(self.L):
            # Stage 2: Anchor scoring
            scores, v = self.scorer(u, memory, prev_selected=cumulative_y)

            # Stage 3: Differentiable selection
            y = self.selector(scores, hard=hard)  # [B, T]

            # Stage 4: Content summary
            z = torch.bmm(y.unsqueeze(1), v).squeeze(1)  # [B, d_u]

            # Stage 5: Selection-history statistics
            c = self.statistics(y, u)  # [B, 4]

            # Stage 6: Memory update
            if feedback is not None:
                r_ell = feedback  # [B, n_feedback] — same for all stages
            else:
                r_ell = torch.zeros(B, self.feedback_dim, device=device, dtype=x.dtype)

            gru_input = torch.cat([z, c, r_ell], dim=-1)  # [B, d_u + 4 + n_feedback]
            memory = self.gru(gru_input, memory)  # [B, d_m]

            # Update cumulative selection for coverage bias
            cumulative_y = cumulative_y + y

            # Store outputs
            all_y.append(y)
            all_z.append(z)
            all_memory.append(memory)
            # Anchor tokens weighted by selection
            anchors = u * y.unsqueeze(-1)  # [B, T, d_u]
            all_anchors.append(anchors)

        all_y = torch.stack(all_y, dim=1)  # [B, L, T]
        all_z = torch.stack(all_z, dim=1)  # [B, L, d_u]
        all_memory = torch.stack(all_memory, dim=1)  # [B, L+1, d_m]
        all_anchors = torch.stack(all_anchors, dim=1)  # [B, L, T, d_u]

        return all_y, all_z, all_memory, all_anchors

    def get_selected_indices(self, y: Tensor) -> Tensor:
        """Extract hard selected indices from selection weights.

        Parameters
        ----------
        y : Tensor  [B, T]
            Selection weights (one routing stage).

        Returns
        -------
        indices : Tensor  [B, K]
            Top-K selected position indices, padded with -1 if fewer than K.
        """
        B, T = y.shape
        K_eff = min(self.K, T)
        _, top_idx = torch.topk(y, K_eff, dim=1)  # [B, K_eff]

        if K_eff < self.K:
            padding = torch.full(
                (B, self.K - K_eff), -1, device=y.device, dtype=top_idx.dtype
            )
            top_idx = torch.cat([top_idx, padding], dim=1)

        return top_idx
