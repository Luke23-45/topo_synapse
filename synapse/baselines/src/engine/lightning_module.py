"""BaselineLightningModule — Lightning wrapper for Z3UnifiedModel.

Mirrors ``SynapseLightningModule`` but wraps ``Z3UnifiedModel`` so
baseline backbones (MLP, TCN, PTv3, SNN, Deep Hodge) can be trained
with the full Lightning Trainer infrastructure.

Features (mirroring legacy baselines):
    - Optimizer: AdamW with optional fused CUDA kernels
    - LR schedule: Linear warmup + cosine decay
    - Gradient clipping: max_norm via Trainer
    - Mixed precision: Optional AMP (bf16/fp16)
    - EMA: Exponential moving average shadow model (via callback)
    - Checkpointing: Best + last model checkpoints
    - Logging: Per-step train metrics, per-epoch val/test metrics
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from synapse.synapse_arch.unified import Z3UnifiedModel
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

class BaselineLightningModule(pl.LightningModule):
    """LightningModule wrapping Z3UnifiedModel for baseline training.

    Parameters
    ----------
    model : Z3UnifiedModel
        The instantiated baseline model.
    loss_config : LossConfig
        Loss weighting configuration.
    lr : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    beta1 : float
        AdamW beta1.
    beta2 : float
        AdamW beta2.
    warmup_steps : int
        Number of linear warmup steps.
    gradient_clip_norm : float
        Max gradient norm for clipping.
    fused_adamw : bool
        Use fused AdamW on CUDA.
    compile_model : bool
        Apply torch.compile to the model.
    """

    def __init__(
        self,
        model: Z3UnifiedModel,
        loss_config: LossConfig | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_steps: int = 500,
        gradient_clip_norm: float = 1.0,
        fused_adamw: bool = True,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Optional torch.compile
        if compile_model and hasattr(torch, "compile"):
            log.info("Applying torch.compile to baseline model")
            model = torch.compile(model)

        self.model = model
        self.loss_config = loss_config or LossConfig()
        self._lr = lr
        self._weight_decay = weight_decay
        self._beta1 = beta1
        self._beta2 = beta2
        self._warmup_steps = warmup_steps
        self._gradient_clip_norm = gradient_clip_norm
        self._fused_adamw = fused_adamw

        # Store full ramp config so we can compute per-epoch weights
        self._base_proxy_weight = self.loss_config.proxy_weight
        self._base_sparsity_weight = self.loss_config.sparsity_weight
        self._aux_ramp_start = self.loss_config.aux_ramp_start
        self._aux_ramp_end = self.loss_config.aux_ramp_end

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None):
        return self.model(sequence, mask)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Apply auxiliary-loss ramp at the start of each epoch.

        The ramp linearly scales ``proxy_weight`` and ``sparsity_weight``
        from 0 to their configured base values over the epoch range
        ``[aux_ramp_start, aux_ramp_end]``.  Before the ramp starts both
        weights are zero; after the ramp ends they hold at their full
        configured values.
        """
        epoch = self.current_epoch
        ramp_start = self._aux_ramp_start
        ramp_end = self._aux_ramp_end

        if epoch < ramp_start:
            scale = 0.0
        elif epoch >= ramp_end:
            scale = 1.0
        else:
            span = max(ramp_end - ramp_start, 1)
            scale = (epoch - ramp_start) / span

        self.loss_config = LossConfig(
            proxy_weight=self._base_proxy_weight * scale,
            sparsity_weight=self._base_sparsity_weight * scale,
            aux_ramp_start=ramp_start,
            aux_ramp_end=ramp_end,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        sequence = batch["sequences"]
        out = self.model(sequence)
        loss, parts = compute_loss(out, batch, self.loss_config)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=sequence.shape[0])
        for key, value in parts.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True,
                     batch_size=sequence.shape[0])

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

        self.log("val/loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=sequence.shape[0])
        self.log("val/accuracy", acc, prog_bar=True, on_epoch=True,
                 batch_size=sequence.shape[0])
        for key, value in parts.items():
            self.log(f"val/{key}", value, on_epoch=True,
                     batch_size=sequence.shape[0])

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
            self.log(f"test/{key}", value, on_epoch=True,
                     batch_size=sequence.shape[0])

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with linear warmup + cosine decay."""
        kwargs: dict[str, Any] = {
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "betas": (self._beta1, self._beta2),
        }
        # Fused AdamW on CUDA (PyTorch 2.0+)
        if self._fused_adamw and self.trainer.accelerator == "gpu":
            try:
                kwargs["fused"] = True
            except (TypeError, AttributeError):
                pass

        optimizer = AdamW(self._traversable_model.parameters(), **kwargs)

        # Estimate total training steps
        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
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
    def _traversable_model(self) -> torch.nn.Module:
        """Return the underlying model for optimizer param groups.

        If torch.compile was used, return the original model so that
        parameter iteration works correctly.
        """
        return getattr(self.model, "_orig_mod", self.model)

    # ------------------------------------------------------------------
    # Gradient clipping (delegated to Trainer)
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip gradients before optimizer step."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self._gradient_clip_norm,
            gradient_clip_algorithm="norm",
        )


__all__ = ["BaselineLightningModule"]
