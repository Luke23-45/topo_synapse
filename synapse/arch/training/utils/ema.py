"""Exponential Moving Average (EMA) for Z3 SYNAPSE training.

When using the Lightning pipeline, EMA is handled by the
``EMACallback``.  This module provides a standalone EMA class for
scripts that don't use Lightning, mirroring the baselines EMAModel.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class ExponentialMovingAverage:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of the model weights that is updated as an
    exponential moving average of the live training weights.  At
    evaluation time, the EMA weights can be swapped in via
    ``apply_to()`` and restored with ``restore()``.

    Parameters
    ----------
    model : nn.Module
        The model to track.
    decay : float
        EMA decay factor (0.999 = slow average, 0.9 = fast).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.decay = decay
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with EMA of current model weights."""
        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def apply_to(self, model: nn.Module) -> None:
        """Swap EMA shadow weights into the model (for evaluation).

        Call ``restore()`` after evaluation to put the original
        training weights back.
        """
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        model.load_state_dict(
            {name: param for name, param in self.shadow.state_dict().items()
             if name in dict(model.named_parameters())},
            strict=False,
        )

    def restore(self, model: nn.Module) -> None:
        """Restore the original training weights after EMA evaluation."""
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load shadow model state dict."""
        self.shadow.load_state_dict(state_dict)
