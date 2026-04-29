"""Core configuration for Z3 baseline experiments.

Only contains what is NEW — the experiment orchestration config.
All data, normalization, and training infrastructure is reused from:
    - ``synapse.synapse.data``           — data pipeline
    - ``synapse.synapse.training``       — Lightning training engine
    - ``synapse.synapse.losses``         — loss functions
    - ``synapse.synapse.training.builders`` — model construction + normalization
"""

from .config import (
    BackboneCondition,
    Z3DatasetSpec,
    ModelParams,
    TrainingParams,
    LossParams,
    StatsParams,
    RolloutParams,
    Z3ExperimentConfig,
    load_experiment_config,
)

__all__ = [
    "BackboneCondition",
    "Z3DatasetSpec",
    "ModelParams",
    "TrainingParams",
    "LossParams",
    "StatsParams",
    "RolloutParams",
    "Z3ExperimentConfig",
    "load_experiment_config",
]
