"""Geometric lift helpers for both legacy and structure-aware paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .topology_features import build_structural_feature_tensor


@dataclass
class Anchor:
    """Legacy exact-audit anchor representation."""

    t: float
    s: np.ndarray
    delta: int
    xi: float
    index: int


def topology_project_numpy(vectors: np.ndarray) -> np.ndarray:
    """Legacy topology projection that removes absolute time."""
    projected = np.array(vectors, copy=True)
    if projected.size > 0:
        projected[:, 0] = 0.0
    return projected


def topology_project_torch(vectors: torch.Tensor) -> torch.Tensor:
    """Legacy topology projection that removes absolute time."""
    projected = vectors.clone()
    if projected.numel() > 0:
        projected[..., 0] = 0.0
    return projected


def dense_anchor_vectors(sequence: torch.Tensor, saliency_scores: torch.Tensor) -> torch.Tensor:
    """Build dense structure-aware vectors for the differentiable lift."""
    return build_structural_feature_tensor(
        sequence,
        selection_weights=saliency_scores,
        include_selection=False,
    )


def anchor_vectors(anchors: List[Anchor], D: Optional[int] = None) -> np.ndarray:
    """Build the legacy exact-audit anchor matrix."""
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
        V[j, 1 : 1 + d] = a.s
        V[j, 1 + d] = a.delta
        V[j, 2 + d] = a.xi

    return V


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


def apply_lift(V_norm: np.ndarray, W_Theta: np.ndarray) -> np.ndarray:
    """Apply the learned affine lift matrix."""
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
