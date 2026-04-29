"""LR scheduler utilities for Z3 SYNAPSE training.

When using the Lightning training pipeline, the scheduler is configured
inside ``SynapseLightningModule.configure_optimizers()``.  This module
provides the same scheduler as a standalone utility for backward
compatibility and for scripts that don't use Lightning.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, epochs: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine annealing scheduler (legacy compatibility).

    For new code, prefer ``build_cosine_warmup_scheduler()`` which
    includes a linear warmup phase, matching the baselines pattern.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 500,
    total_steps: int = 10000,
) -> LambdaLR:
    """Linear warmup + cosine decay LR schedule.

    This is the same schedule used by the baselines Trainer and by
    ``SynapseLightningModule.configure_optimizers()``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_steps : int
        Number of linear warmup steps.
    total_steps : int
        Total number of training steps.

    Returns
    -------
    LambdaLR
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

    return LambdaLR(optimizer, lr_lambda)
