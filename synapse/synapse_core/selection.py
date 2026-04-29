"""
Anchor Selection — Z3 Reference: §5–6 of 01_main_definition.md

Implements:
    - Relaxed selector: y* = argmax_{y ∈ Π_{K,r,T}} (s^T y − λ||y||₂²)
    - Hard projection: I* = Sel_{K,r,T}(y*)
    - Anchor construction: A* from I*

Note: The relaxed selector delegates to the root synapse_core.anchor_selector
which contains the battle-tested OSQP/CVXPY QP solver. This is intentional —
the QP solver is a well-tested mathematical component that should not be
reimplemented.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .lift import Anchor

# Solver zero tolerance — QP solvers leave small positive artifacts
_SOLVER_ZERO_TOL: float = 1e-4


def solve_relaxed_selector(
    saliency: np.ndarray,
    K: int,
    r: int,
    lam: float,
    solver: str = "osqp",
) -> np.ndarray:
    """Solve the Z3 relaxed selector QP over polytope Π_{K,r,T}.

    Delegates to the root synapse_core.anchor_selector for the actual QP solve.
    """
    from .anchor_selector import solve_relaxed_selector as _solve
    return _solve(saliency, K=K, r=r, lam=lam, solver=solver)


def hard_select_indices(y_star: np.ndarray, K: int, r: int) -> List[int]:
    """Deterministic hard projection Sel_{K,r,T}(y*).

    Algorithm (§6):
        1. Compute supp+(y*) = {t : y*_t > 0}
        2. Sort by decreasing y*_t, ties by smaller t
        3. Greedily accept indices satisfying |i-j| > r
    """
    T = len(y_star)
    if T < 2 or K < 1:
        return []

    positive_support = [t for t in range(1, T) if y_star[t] > _SOLVER_ZERO_TOL]
    if not positive_support:
        return []

    sorted_indices = sorted(positive_support, key=lambda t: (-y_star[t], t))

    retained = []
    for idx in sorted_indices:
        if len(retained) >= K:
            break
        if all(abs(idx - ret) > r for ret in retained):
            retained.append(idx)

    retained.sort()
    return retained


def build_anchor_sequence(indices: List[int], trajectory: np.ndarray, event_scores: np.ndarray) -> List[Anchor]:
    """Build anchor sequence A* from retained index set I*.

    For each i_j ∈ I*:
        t_j = (i_j + 1) / T
        s_j = x_{i_j}
        δ_1 = i_1, δ_j = i_j - i_{j-1}
        ξ_j = e_{i_j}
    """
    T = trajectory.shape[0]
    if not indices:
        return []

    anchors = []
    for j, idx in enumerate(indices):
        t_j = (idx + 1) / T
        s_j = trajectory[idx].copy()
        delta_j = idx if j == 0 else idx - indices[j - 1]
        xi_j = float(event_scores[idx])
        anchors.append(Anchor(t=t_j, s=s_j, delta=delta_j, xi=xi_j, index=idx))

    return anchors


__all__ = [
    "build_anchor_sequence",
    "hard_select_indices",
    "solve_relaxed_selector",
]
