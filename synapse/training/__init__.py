"""Training utilities for Z3 end-to-end experiments.

Submodules
----------
core
    Lightning-based training loop, LightningModule, and LightningDataModule.
builders
    Model construction from config + normalization resolution.
evaluation
    Post-training evaluation hooks + evaluator factory.
utils
    Checkpoint, scheduler, EMA, and Lightning callbacks.
trainer
    Backward-compatible re-exports of ``train`` and ``build_model_from_cfg``.
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
