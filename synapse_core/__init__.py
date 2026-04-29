"""Z3-aligned mathematical core.

Exports:
    - DifferentiableHodgeProxy: full Hodge L0+L1 spectral proxy (§12)
    - compute_exact_topology_audit: deployment-time Vietoris-Rips audit (§9)
    - CausalEventModel: event detection + saliency normalization (§4)
    - Selection functions: relaxed selector + hard projection (§5–6)
    - Lift functions: topology projection + normalization + learned lift (§7–9)
    - Topology functions: exact persistence + Hausdorff distance (§9)
"""

from .audit import compute_exact_topology_audit
from .event import CausalEventModel
from .lift import (
    Anchor,
    anchor_vectors,
    apply_lift,
    dense_anchor_vectors,
    normalize_anchors,
    topology_project_numpy,
    topology_project_torch,
)
from .proxy import DifferentiableHodgeProxy
from .selection import build_anchor_sequence, hard_select_indices, solve_relaxed_selector
from .topology import compute_persistence_diagrams, hausdorff_distance, summarize_diagrams

__all__ = [
    "Anchor",
    "CausalEventModel",
    "DifferentiableHodgeProxy",
    "anchor_vectors",
    "apply_lift",
    "build_anchor_sequence",
    "compute_exact_topology_audit",
    "compute_persistence_diagrams",
    "dense_anchor_vectors",
    "hard_select_indices",
    "hausdorff_distance",
    "normalize_anchors",
    "solve_relaxed_selector",
    "summarize_diagrams",
    "topology_project_numpy",
    "topology_project_torch",
]
