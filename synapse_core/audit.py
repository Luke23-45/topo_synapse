"""
Exact Topology Audit — Z3 Reference: §9 of 01_main_definition.md

Computes the deployment-time exact topological object:

    T^exact_Θ(x_{1:T}) = (A*, P^top_Θ, Dgm_0, ..., Dgm_Q)

This is the "ground truth" topological analysis used for validation,
NOT the differentiable proxy used during training.
"""

from __future__ import annotations

import numpy as np
import torch

from synapse.common.types import ExactTopologyAudit

from .lift import (
    Anchor,
    anchor_vectors,
    apply_lift,
    dense_anchor_vectors,
    normalize_anchors,
    topology_project_numpy,
)
from .selection import build_anchor_sequence, hard_select_indices
from .topology import compute_persistence_diagrams, summarize_diagrams


def compute_exact_topology_audit(
    trajectory: np.ndarray,
    event_scores: np.ndarray,
    saliency_scores: np.ndarray,
    y_star: np.ndarray,
    W_theta: np.ndarray,
    *,
    K: int,
    r: int,
    Q: int,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> ExactTopologyAudit:
    """Full deployment-time exact topology audit pipeline.

    Steps:
        1. Hard-select anchor indices from y*
        2. Build anchor sequence A*
        3. Construct anchor vectors v(a_j)
        4. Apply topology projection Π_top (zero out time)
        5. Normalize with N(v)
        6. Apply learned lift W_Θ
        7. Compute exact persistence diagrams via Vietoris-Rips
    """
    indices = hard_select_indices(y_star, K=K, r=r)
    anchors = build_anchor_sequence(indices, trajectory, event_scores)
    anchor_matrix = anchor_vectors(anchors, D=trajectory.shape[1] + 3)
    projected = topology_project_numpy(anchor_matrix)
    normalized, _, _ = normalize_anchors(projected, mu=mu, sigma=sigma)
    cloud = apply_lift(normalized, W_theta)
    diagrams = compute_persistence_diagrams(cloud, Q=Q)
    summary = summarize_diagrams(diagrams)
    return ExactTopologyAudit(
        anchor_indices=indices,
        anchor_vectors=projected,
        normalized_anchor_vectors=normalized,
        point_cloud=cloud,
        persistence_diagrams=diagrams,
        topology_summary=summary,
        event_scores=event_scores,
        saliency_scores=saliency_scores,
        y_star=y_star,
    )


def dense_proxy_vectors_for_sequence(
    sequence: torch.Tensor,
    saliency_scores: torch.Tensor,
) -> torch.Tensor:
    """Build dense training-time proxy vectors from a sequence tensor."""
    return dense_anchor_vectors(sequence, saliency_scores)
