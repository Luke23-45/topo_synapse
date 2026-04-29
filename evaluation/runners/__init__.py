"""Evaluation runners for Z3 SYNAPSE models.

Each runner implements the evaluation protocol for a specific data
modality / task type, producing a standardized ``EvalResult`` that
feeds into the reporting and visualization pipeline.

Submodules
----------
base
    Abstract ``BaseEvaluator`` and the shared ``EvalResult`` dataclass.
classification
    Classification task evaluator (temporal, geometric, scientific modalities).
temporal
    Temporal-sequence specific evaluation (noise robustness, length scaling).
geometric
    Geometric / point-cloud specific evaluation (rotation invariance, density).
scientific
    Scientific-data specific evaluation (feature importance, regime detection).
"""

from .base import BaseEvaluator, EvalResult
from .classification import ClassificationEvaluator

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "ClassificationEvaluator",
]
