"""Clean tqdm progress bar for Z3 Baseline training.

Replaces Lightning's default verbose progress bar with a minimal tqdm
display that only prints metrics at epoch boundaries.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from tqdm.auto import tqdm


class EpochTqdmProgressBar(pl.callbacks.ProgressBar):
    """Custom progress bar that shows a single tqdm bar per epoch
    and only prints metric summaries at epoch transitions.

    - During training: a single tqdm bar per epoch (no per-step metric spam).
    - At epoch end: one clean line with train_loss, val_loss, val_accuracy.
    - Validation sanity check is suppressed from the bar.
    """

    def __init__(self) -> None:
        super().__init__()
        self._train_bar: tqdm | None = None
        self._val_bar: tqdm | None = None
        self._is_sanity: bool = False

    # -- Disable Lightning's default bars ---------------------------------

    def init_train_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_validation_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_predict_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_test_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    # -- Sanity check handling --------------------------------------------

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._is_sanity = True

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._is_sanity = False

    # -- Training epoch ---------------------------------------------------

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_sanity:
            return
        total = self.total_train_batches
        desc = f"Epoch {trainer.current_epoch}/{trainer.max_epochs - 1}"
        self._train_bar = tqdm(
            total=total,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._train_bar is not None and not self._train_bar.disable:
            self._train_bar.update(1)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._train_bar is not None:
            self._train_bar.close()
            self._train_bar = None

    # -- Validation epoch -------------------------------------------------

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_sanity:
            return
        total = self.total_val_batches
        self._val_bar = tqdm(
            total=total,
            desc="  Validating",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self._val_bar is not None and not self._val_bar.disable:
            self._val_bar.update(1)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._val_bar is not None:
            self._val_bar.close()
            self._val_bar = None

        if self._is_sanity:
            return

        # Print one clean summary line at epoch transition
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = metrics.get("train/loss_epoch")
        val_loss = metrics.get("val/loss")
        val_acc = metrics.get("val/accuracy")

        parts = [f"Epoch {epoch}"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_acc is not None:
            parts.append(f"val_acc={val_acc:.4f}")

        tqdm.write("  │ ".join(parts))

    # -- Required property overrides --------------------------------------

    @property
    def train_batch_bar(self) -> tqdm:
        return self._train_bar or tqdm(disable=True)

    @property
    def val_batch_bar(self) -> tqdm:
        return self._val_bar or tqdm(disable=True)

    @property
    def test_batch_bar(self) -> tqdm:
        return tqdm(disable=True)

    @property
    def predict_batch_bar(self) -> tqdm:
        return tqdm(disable=True)


__all__ = ["EpochTqdmProgressBar"]
