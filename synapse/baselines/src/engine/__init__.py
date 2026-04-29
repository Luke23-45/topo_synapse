"""Z3 Baseline Engine — training, evaluation, rollout, and statistical testing.

Training delegates to ``synapse.synapse.training`` (Lightning).
Evaluation, rollout, and metrics are baseline-study-specific (classification).
"""

from .evaluate import BackboneEvaluation, evaluate_backbone
from .metrics import ComparisonResult, compare_conditions
from .rollout import RolloutResult, rollout_evaluate, rollout_evaluate_dataset, aggregate_rollout_results

__all__ = [
    "BackboneEvaluation",
    "evaluate_backbone",
    "ComparisonResult",
    "compare_conditions",
    "RolloutResult",
    "rollout_evaluate",
    "rollout_evaluate_dataset",
    "aggregate_rollout_results",
]
