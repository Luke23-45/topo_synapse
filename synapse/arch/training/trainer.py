"""Training module — backward-compatible re-exports.

The training logic has been modularized into:
    - ``builders``: Model construction + normalization resolution
    - ``core``: Lightning training loop, LightningModule, DataModule
    - ``evaluation``: Post-training evaluation + evaluator factory
    - ``utils``: Checkpoint, scheduler, EMA, and Lightning callbacks

This file re-exports ``train`` and ``build_model_from_cfg`` for
backward compatibility with existing imports.
"""

from .builders import build_model_from_cfg, resolve_normalization
from .core import SynapseDataModule, SynapseLightningModule, train
from .evaluation import build_evaluator, run_post_training_evaluation
from .utils import EMACallback, PostTrainEvalCallback

__all__ = [
    "EMACallback",
    "PostTrainEvalCallback",
    "SynapseDataModule",
    "SynapseLightningModule",
    "build_evaluator",
    "build_model_from_cfg",
    "resolve_normalization",
    "run_post_training_evaluation",
    "train",
]
