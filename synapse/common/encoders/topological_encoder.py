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
import torch.nn.functional as F
from torch import nn, Tensor

from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.lift import (
    dense_anchor_vectors,
    topology_project_torch,
)
from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.synapse_arch.normalized_lift import NormalizedLift


class SoftSelectorProxy(nn.Module):
    """Differentiable GPU proxy for the relaxed selector.

    Replaces the per-sample CPU QP solve with a fully GPU-based
    approximation using sigmoid relaxation of budget + refractory
    constraints.  Gradients flow through the proxy, enabling
    end-to-end training without GPU→CPU sync barriers.

    The proxy approximates the QP:
        minimize  λ Σ y² - Σ s_t y_t
        s.t.  Σ y_t ≤ K,  y_t + y_u ≤ 1 for |t-u| ≤ r,  0 ≤ y ≤ 1
    """

    def __init__(self, K: int, r: int, lam: float, init_temperature: float = 1.0):
        super().__init__()
        self.K = K
        self.r = r
        self.lam = lam
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    def forward(self, saliency: Tensor) -> Tensor:
        B, T = saliency.shape
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)

        # Shift logits by saliency so high-saliency positions are preferred
        logits = (saliency / (2.0 * self.lam) - 0.5) / temp
        y = torch.sigmoid(logits)

        # Enforce y[:, 0] = 0 (first-step constraint)
        y = y.clone()
        y[:, 0] = 0.0

        # Soft budget: scale down if sum exceeds K
        budget = y.sum(dim=1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.K / budget, max=1.0)
        y = y * scale

        # Soft refractory: for each pair |t-u| ≤ r, dampen overlap
        for d in range(1, min(self.r + 1, T)):
            shift = torch.roll(y, shifts=-d, dims=1)
            pair_sum = y + shift
            damping = torch.clamp(2.0 / (1.0 + pair_sum), max=1.0)
            y = y * damping

        y[:, 0] = 0.0
        return y


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

        # Stage 2: Anchor selection (GPU proxy)
        self.selector_proxy = SoftSelectorProxy(K=K, r=r, lam=lam)

        # Stage 3: Geometric lift
        self.lift = NormalizedLift(anchor_dim, k)

        # Stage 4: Project lifted k-dim to d_model
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

    def _gpu_selector(self, saliency_scores: Tensor) -> Tensor:
        """Differentiable GPU proxy for anchor selection."""
        return self.selector_proxy(saliency_scores)

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

        # Stage 2: Anchor selection (GPU proxy — differentiable, no CPU sync)
        y_star = self._gpu_selector(saliency_scores)

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
