"""Z3 SYNAPSE Evaluation Module.

Provides a modular evaluation framework for Z3 topology-first models,
organized by concern:

    - **metrics/**: Classification, statistical, and topology metrics
    - **runners/**: Modality-specific evaluators (temporal, geometric, scientific)
    - **configs/**: YAML evaluation configurations per dataset
    - **visualization/**: Publication-quality plots and topology visualizations
    - **reporting/**: JSON and Markdown report generation
    - **rollout**: Autoregressive robustness evaluation
    - **ablations**: Ablation configuration builder

Quick start::

    from synapse.evaluation.runners import ClassificationEvaluator
    from synapse.evaluation.reporting import generate_json_report

    evaluator = ClassificationEvaluator(config, test_loader, output_dir)
    result = evaluator.evaluate(model)
    generate_json_report({"default": result}, "report.json")
"""

from .ablations import build_ablation_configs
from .metrics import (
    ComparisonResult,
    classification_accuracy,
    compare_conditions,
    per_class_accuracy,
    proxy_exact_alignment,
    top_k_accuracy,
)
from .reporting import generate_json_report, generate_markdown_report
from .rollout import (
    RolloutResult,
    aggregate_rollout_results,
    rollout_evaluate,
    rollout_evaluate_dataset,
)
from .runners import BaseEvaluator, ClassificationEvaluator, EvalResult
from .visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_persistence_diagram,
    plot_point_cloud,
    plot_proxy_exact_scatter,
    plot_robustness_sweep,
)

__all__ = [
    # Ablations
    "build_ablation_configs",
    # Metrics
    "ComparisonResult",
    "classification_accuracy",
    "compare_conditions",
    "per_class_accuracy",
    "proxy_exact_alignment",
    "top_k_accuracy",
    # Reporting
    "generate_json_report",
    "generate_markdown_report",
    # Rollout
    "RolloutResult",
    "aggregate_rollout_results",
    "rollout_evaluate",
    "rollout_evaluate_dataset",
    # Runners
    "BaseEvaluator",
    "ClassificationEvaluator",
    "EvalResult",
    # Visualization
    "plot_accuracy_comparison",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_learning_curves",
    "plot_persistence_diagram",
    "plot_point_cloud",
    "plot_proxy_exact_scatter",
    "plot_robustness_sweep",
]

