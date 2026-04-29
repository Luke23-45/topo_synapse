"""Visualization module for Z3 SYNAPSE evaluation.

Generates publication-quality figures from evaluation results.

Submodules
----------
plots
    Standard metric plots (confusion matrix, learning curves, bar charts).
topology_plots
    Topology-specific visualizations (persistence diagrams, point clouds).
"""

from .plots import (
    plot_confusion_matrix,
    plot_learning_curves,
    plot_accuracy_comparison,
    plot_robustness_sweep,
    plot_feature_importance,
)
from .topology_plots import (
    plot_point_cloud,
    plot_persistence_diagram,
    plot_proxy_exact_scatter,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_learning_curves",
    "plot_accuracy_comparison",
    "plot_robustness_sweep",
    "plot_feature_importance",
    "plot_point_cloud",
    "plot_persistence_diagram",
    "plot_proxy_exact_scatter",
]
