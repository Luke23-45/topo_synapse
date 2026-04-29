"""Evaluation metrics for Z3 SYNAPSE models.

Submodules
----------
classification
    Task-level classification metrics (accuracy, per-class, top-k, confusion).
statistical
    Statistical comparison tests (Welch's t-test, Cohen's d, bootstrap CI).
topology
    Topology-specific metrics (proxy-exact alignment, Betti accuracy).
"""

from .classification import (
    classification_accuracy,
    per_class_accuracy,
    top_k_accuracy,
    confusion_matrix_metrics,
)
from .statistical import (
    ComparisonResult,
    cohens_d,
    bootstrap_ci_cohens_d,
    welch_t_test,
    shapiro_wilk_test,
    wilcoxon_test,
    compare_conditions,
)
from .topology import (
    proxy_exact_alignment,
    betti_number_accuracy,
    persistence_diagram_distance,
)

__all__ = [
    "classification_accuracy",
    "per_class_accuracy",
    "top_k_accuracy",
    "confusion_matrix_metrics",
    "ComparisonResult",
    "cohens_d",
    "bootstrap_ci_cohens_d",
    "welch_t_test",
    "shapiro_wilk_test",
    "wilcoxon_test",
    "compare_conditions",
    "proxy_exact_alignment",
    "betti_number_accuracy",
    "persistence_diagram_distance",
]
