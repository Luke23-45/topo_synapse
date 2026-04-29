"""End-to-end runnable scripts.

Each subpackage is a self-contained CLI entry point:

    - ``train``: End-to-end training with post-training evaluation
    - ``eval``: Standalone model evaluation
    - ``infer``: Single-batch inference
    - ``deploy``: Deployment audit test
    - ``ablations``: Ablation experiment runner
"""

from .train.train import main as train_main
from .eval.eval import main as eval_main
from .infer.infer import main as infer_main
from .deploy.deploy import main as deploy_main
from .ablations.ablations import main as ablations_main

__all__ = [
    "train_main",
    "eval_main",
    "infer_main",
    "deploy_main",
    "ablations_main",
]
