from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_loss import classification_loss
from .auxiliary_losses import proxy_regularization, selector_sparsity


@dataclass
class LossConfig:
    proxy_weight: float = 1e-3
    sparsity_weight: float = 1e-4
    aux_ramp_start: int = 0
    aux_ramp_end: int = 5


def compute_loss(output, batch, cfg: LossConfig) -> tuple[torch.Tensor, dict[str, float]]:
    task = classification_loss(output.logits, batch["targets"])
    proxy = proxy_regularization(output.proxy_features)
    sparse = selector_sparsity(output.y_star)
    loss = task + cfg.proxy_weight * proxy + cfg.sparsity_weight * sparse
    return loss, {"task_loss": float(task.item()), "proxy_reg": float(proxy.item()), "selector_sparsity": float(sparse.item())}
