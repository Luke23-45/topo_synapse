"""Clean progress callback for Z3 Baseline training.

No tqdm bars, only one plain-ASCII line per epoch with key metrics.
This keeps notebook and terminal output small and avoids Windows
console encoding failures from box-drawing characters.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class EpochOnlyProgressBar(pl.callbacks.ProgressBar):
    """Zero-spam progress callback."""

    def __init__(self) -> None:
        super().__init__()
        self._is_sanity: bool = False

    def init_train_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_validation_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_predict_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def init_test_tqdm(self) -> tqdm:
        return tqdm(disable=True)

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._is_sanity = True

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._is_sanity = False

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_sanity:
            return

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

        print(" | ".join(parts))

    @property
    def train_batch_bar(self) -> tqdm:
        return tqdm(disable=True)

    @property
    def val_batch_bar(self) -> tqdm:
        return tqdm(disable=True)

    @property
    def test_batch_bar(self) -> tqdm:
        return tqdm(disable=True)

    @property
    def predict_batch_bar(self) -> tqdm:
        return tqdm(disable=True)


__all__ = ["EpochOnlyProgressBar"]
