"""Synapse LightningModule — active topology-first training wrapper.

Encapsulates the active TopologyFirstModel, loss computation, optimizer
configuration, and metric logging in a single LightningModule so the
entire training lifecycle (forward, backward, schedulers, checkpointing,
mixed-precision, gradient clipping, EMA) is managed by the Lightning
Trainer.

Key design decisions (mirroring baselines/engine/train.py):
    - Optimizer: AdamW with optional fused CUDA kernels
    - LR schedule: Linear warmup + cosine decay
    - Gradient clipping: max_norm=1.0
    - Mixed precision: Optional AMP (bf16 on Ampere+, fp16 fallback)
    - EMA: Exponential moving average shadow model (via callback)
    - Logging: Per-step train metrics, per-epoch val/test metrics
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import TopologyFirstModel
from synapse.arch.losses.combined_loss import LossConfig, compute_loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a linear warmup + cosine decay LR schedule."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class SynapseLightningModule(pl.LightningModule):
    """LightningModule wrapping the active topology-first model for training.

    Parameters
    ----------
    model : TopologyFirstModel
        The instantiated active model (with normalization already set).
    config : SynapseConfig
        Architecture config (used for LR, weight decay, etc.).
    loss_config : LossConfig
        Loss weighting configuration.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    warmup_steps : int
        Number of linear warmup steps.
    gradient_clip_norm : float
        Max gradient norm for clipping.
    """

    def __init__(
        self,
        model: TopologyFirstModel,
        config: SynapseConfig,
        loss_config: LossConfig | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.config = config
        self.loss_config = loss_config or LossConfig()
        self._lr = lr
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._gradient_clip_norm = gradient_clip_norm

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None):
        return self.model(sequence, mask)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        sequence = batch["sequences"]
        out = self.model(sequence)
        loss, parts = compute_loss(out, batch, self.loss_config)

        # Log per-step metrics (on_step=True for progress bar, on_epoch for epoch avg)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=sequence.shape[0])
        for key, value in parts.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True, batch_size=sequence.shape[0])

        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        sequence = batch["sequences"]
        out = self.model(sequence)
        loss, parts = compute_loss(out, batch, self.loss_config)

        pred = out.logits.argmax(dim=-1)
        acc = (pred == batch["targets"]).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=sequence.shape[0])
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True, batch_size=sequence.shape[0])
        for key, value in parts.items():
            self.log(f"val/{key}", value, on_epoch=True, batch_size=sequence.shape[0])

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        sequence = batch["sequences"]
        out = self.model(sequence)
        loss, parts = compute_loss(out, batch, self.loss_config)

        pred = out.logits.argmax(dim=-1)
        acc = (pred == batch["targets"]).float().mean()

        self.log("test/loss", loss, on_epoch=True, batch_size=sequence.shape[0])
        self.log("test/accuracy", acc, on_epoch=True, batch_size=sequence.shape[0])
        for key, value in parts.items():
            self.log(f"test/{key}", value, on_epoch=True, batch_size=sequence.shape[0])

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with linear warmup + cosine decay."""
        # Fused AdamW on CUDA (PyTorch 2.0+)
        kwargs: dict[str, Any] = {
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "betas": (0.9, 0.999),
        }
        if self.trainer.accelerator == "gpu":
            try:
                kwargs["fused"] = True
            except (TypeError, AttributeError):
                pass

        optimizer = AdamW(self.traversable_model.parameters(), **kwargs)

        # Estimate total training steps from datamodule or trainer
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback: epochs * steps_per_epoch
            train_loader = self.trainer.train_dataloader
            if train_loader is not None:
                steps_per_epoch = len(train_loader)
            else:
                steps_per_epoch = 1
            total_steps = self.trainer.max_epochs * steps_per_epoch

        scheduler = _cosine_warmup_scheduler(
            optimizer,
            warmup_steps=self._warmup_steps,
            total_steps=int(total_steps),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @property
    def traversable_model(self) -> torch.nn.Module:
        """Return the underlying model for optimizer param groups.

        If torch.compile was used, return the original model so that
        parameter iteration works correctly.
        """
        return getattr(self.model, "_orig_mod", self.model)

    # ------------------------------------------------------------------
    # Gradient clipping (delegated to Trainer, but we set the norm)
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip gradients before optimizer step."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self._gradient_clip_norm,
            gradient_clip_algorithm="norm",
        )
