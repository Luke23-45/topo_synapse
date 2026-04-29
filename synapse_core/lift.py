"""
Geometric Lift — Z3 Reference: §8, §10 of 01_main_definition.md

Implements:
    - Topology projection Π_top: zeroes out absolute time coordinate
    - Dense anchor vector construction for training-time proxy
    - Anchor vector construction for deployment-time audit
    - Affine normalization N(v) = D^{-1}(v - μ)
    - Learned lift ρ_Θ(a) = W_Θ · N(Π_top(v(a)))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


# ─── Anchor dataclass ───────────────────────────────────────────────

@dataclass
class Anchor:
    """A single retained anchor: a_j = (t_j, s_j, δ_j, ξ_j)."""
    t: float
    s: np.ndarray
    delta: int
    xi: float
    index: int


# ─── Topology Projection (§8) ───────────────────────────────────────

def topology_project_numpy(vectors: np.ndarray) -> np.ndarray:
    """Π_top: zero out absolute time coordinate (column 0)."""
    projected = np.array(vectors, copy=True)
    if projected.size > 0:
        projected[:, 0] = 0.0
    return projected


def topology_project_torch(vectors: torch.Tensor) -> torch.Tensor:
    """Π_top: zero out absolute time coordinate (column 0)."""
    projected = vectors.clone()
    if projected.numel() > 0:
        projected[..., 0] = 0.0
    return projected


# ─── Dense Anchor Vectors for Training (§10) ────────────────────────

def dense_anchor_vectors(sequence: torch.Tensor, saliency_scores: torch.Tensor) -> torch.Tensor:
    """Build dense per-time surrogate vectors ṽ_t for training proxy.

    ṽ_t = [t̃_t, s̃_t^T, δ̃_t, ξ̃_t]^T ∈ ℝ^{d+3}
    """
    batch, steps, _ = sequence.shape
    time = torch.linspace(
        1.0 / max(steps, 1),
        1.0,
        steps,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    time = time.unsqueeze(0).expand(batch, -1)
    delta = torch.ones_like(time)
    delta[:, 0] = 0.0
    return torch.cat(
        [time.unsqueeze(-1), sequence, delta.unsqueeze(-1), saliency_scores.unsqueeze(-1)],
        dim=-1,
    )


# ─── Deployment Anchor Vector Construction (§7) ─────────────────────

def anchor_vectors(anchors: List[Anchor], D: Optional[int] = None) -> np.ndarray:
    """Build anchor vector matrix V from anchor sequence A*.

    v(a_j) = [t_j, s_j^T, δ_j, ξ_j]^T ∈ ℝ^{d+3}
    """
    if not anchors:
        if D is None:
            return np.empty((0, 0), dtype=np.float64)
        return np.empty((0, D), dtype=np.float64)

    m = len(anchors)
    d = anchors[0].s.shape[0]
    D_actual = d + 3
    V = np.zeros((m, D_actual), dtype=np.float64)

    for j, a in enumerate(anchors):
        V[j, 0] = a.t
        V[j, 1:1 + d] = a.s
        V[j, 1 + d] = a.delta
        V[j, 2 + d] = a.xi

    return V


# ─── Normalization (§8) ─────────────────────────────────────────────

def normalize_anchors(
    V: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply affine normalization N(v) = D^{-1}(v - μ)."""
    if V.shape[0] == 0:
        D = V.shape[1] if V.ndim == 2 and V.shape[1] > 0 else 0
        empty_norm = np.empty((0, D), dtype=np.float64)
        mu_out = np.zeros(D, dtype=np.float64) if D > 0 else np.array([], dtype=np.float64)
        sigma_out = np.ones(D, dtype=np.float64) if D > 0 else np.array([], dtype=np.float64)
        return empty_norm, mu_out, sigma_out

    D = V.shape[1]
    if mu is None:
        mu = np.mean(V, axis=0)
    else:
        mu = np.asarray(mu, dtype=np.float64)
    if sigma is None:
        sigma = np.std(V, axis=0)
        sigma[sigma <= 0] = 1.0
    else:
        sigma = np.asarray(sigma, dtype=np.float64)

    sigma = np.where(sigma <= 0, 1.0, sigma)
    V_norm = (V - mu) / sigma
    return V_norm, mu, sigma


# ─── Learned Lift (§9) ──────────────────────────────────────────────

def apply_lift(V_norm: np.ndarray, W_Theta: np.ndarray) -> np.ndarray:
    """Apply learned lift: ρ_Θ(a_j) = W_Θ · N(v(a_j)) ∈ ℝ^k."""
    if V_norm.shape[0] == 0:
        k = W_Theta.shape[0]
        return np.empty((0, k), dtype=np.float64)
    return (V_norm @ W_Theta.T).astype(np.float64)


__all__ = [
    "Anchor",
    "anchor_vectors",
    "apply_lift",
    "dense_anchor_vectors",
    "normalize_anchors",
    "topology_project_numpy",
    "topology_project_torch",
]
