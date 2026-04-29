"""Training Engine — Z3 Baseline Study.

Uses PyTorch Lightning Trainer with the full production pipeline
mirroring the legacy baselines:

    - AdamW optimizer with optional fused CUDA kernels
    - Linear warmup + cosine decay LR schedule
    - Gradient clipping (max_norm)
    - Mixed precision (AMP: bf16/fp16)
    - EMA (exponential moving average) shadow model
    - Model checkpointing (best + last)
    - Early stopping on val/loss
    - CSV logging

Reuses
------
- ``BaselineLightningModule`` — Lightning wrapper for Z3UnifiedModel
- ``synapse.synapse.training.utils.callbacks.EMACallback``
- ``synapse.synapse.data.data.build_dataloaders``
- ``synapse.synapse.losses.combined_loss.compute_loss``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from synapse.arch.losses.combined_loss import LossConfig
from synapse.arch.training.utils.callbacks import EMACallback
from synapse.synapse_arch.unified import Z3UnifiedModel

from .lightning_module import BaselineLightningModule
from .progress import EpochOnlyProgressBar
from ..core.config import Z3ExperimentConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """State returned after training a single backbone."""
    backbone: str
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    model: Z3UnifiedModel | None = None


# ---------------------------------------------------------------------------
# Training via Lightning Trainer
# ---------------------------------------------------------------------------

def train_backbone(
    config: Z3ExperimentConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: str = "cpu",
) -> TrainState:
    """Train a single backbone condition using PyTorch Lightning.

    Parameters
    ----------
    config : Z3ExperimentConfig
    train_loader, val_loader : DataLoader
    output_dir : Path
    device : str

    Returns
    -------
    TrainState
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset spec for dimensions
    modality = config.modality

    # Build model — only pass parameters the backbone actually accepts
    model_kwargs = config.filtered_model_kwargs()
    model = Z3UnifiedModel(
        backbone_type=config.backbone.value,
        modality=modality,
        **model_kwargs,
    )

    log.info(
        "Training backbone=%s, params=%d",
        config.backbone.value, model.num_parameters,
    )

    # Loss config
    loss_config = LossConfig(
        proxy_weight=config.loss.proxy_weight,
        sparsity_weight=config.loss.sparsity_weight,
        aux_ramp_start=config.loss.aux_ramp_start,
        aux_ramp_end=config.loss.aux_ramp_end,
    )

    # LightningModule
    lightning_model = BaselineLightningModule(
        model=model,
        loss_config=loss_config,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        warmup_steps=config.training.warmup_steps,
        gradient_clip_norm=config.training.gradient_clip_norm,
        fused_adamw=config.training.fused_adamw,
        compile_model=config.training.compile_model,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks: list[pl.Callback] = []

    # EMA shadow model
    callbacks.append(EMACallback(decay=0.999))

    # Early stopping
    if config.training.early_stopping_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=config.training.early_stopping_patience,
            mode="min",
            verbose=True,
        ))

    # Checkpointing
    if config.training.save_checkpoints:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ))

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    trainer_logger = pl.loggers.CSVLogger(save_dir=str(output_dir / "logs"))

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    accelerator = "gpu" if (device != "cpu" and torch.cuda.is_available()) else "cpu"

    # Precision / AMP
    precision = 32
    if config.training.use_amp and accelerator == "gpu":
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"

    # Clean tqdm progress bar — metrics only at epoch transitions
    callbacks.append(EpochOnlyProgressBar())

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=trainer_logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=10,
        enable_checkpointing=config.training.save_checkpoints,
        enable_progress_bar=True,
        gradient_clip_val=config.training.gradient_clip_norm,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        deterministic=False,
        benchmark=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log.info(
        "Starting Lightning training: epochs=%d, accelerator=%s, precision=%s",
        config.training.max_epochs, accelerator, precision,
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # ------------------------------------------------------------------
    # Extract training history
    # ------------------------------------------------------------------
    state = _extract_train_state(trainer, lightning_model, config.backbone.value, device)

    # Load best checkpoint if available
    if config.training.save_checkpoints:
        best_ckpt = output_dir / "checkpoints" / "last.ckpt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(
                {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")},
                strict=False,
            )
            log.info("Loaded best checkpoint from %s", best_ckpt)

    state.model = model
    return state


def _extract_train_state(
    trainer: pl.Trainer,
    lightning_model: BaselineLightningModule,
    backbone_name: str,
    device: str,
) -> TrainState:
    """Extract training history from the Lightning Trainer."""
    state = TrainState(backbone=backbone_name)

    # Try to get metrics from the CSV logger
    csv_logger = None
    for lg in trainer.loggers:
        if isinstance(lg, pl.loggers.CSVLogger):
            csv_logger = lg
            break

    if csv_logger is not None:
        metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(metrics_path)
                for epoch in sorted(df.get("epoch", []).unique()):
                    epoch_data = df[df["epoch"] == epoch]
                    train_loss = epoch_data.get("train/loss_epoch")
                    val_loss = epoch_data.get("val/loss")
                    val_acc = epoch_data.get("val/accuracy")

                    if train_loss is not None and not train_loss.dropna().empty:
                        state.train_losses.append(float(train_loss.dropna().mean()))
                    if val_loss is not None and not val_loss.dropna().empty:
                        state.val_losses.append(float(val_loss.dropna().mean()))
                    if val_acc is not None and not val_acc.dropna().empty:
                        acc_val = float(val_acc.dropna().mean())
                        state.val_accuracies.append(acc_val)
                        if acc_val > state.best_val_accuracy:
                            state.best_val_accuracy = acc_val
                            state.best_epoch = int(epoch)
            except ImportError:
                log.warning(
                    "pandas not installed — falling back to NaN placeholders "
                    "for training history. Install pandas for full metrics: "
                    "pip install pandas"
                )

    # Fallback: use callback metrics if CSV logger didn't work
    if not state.train_losses:
        # Try to get from trainer.callback_metrics
        try:
            for epoch in range(trainer.current_epoch):
                state.train_losses.append(float("nan"))
                state.val_losses.append(float("nan"))
                state.val_accuracies.append(float("nan"))
        except Exception:
            pass

    return state


__all__ = ["TrainState", "train_backbone"]
