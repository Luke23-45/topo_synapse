"""Core training loop for Z3 SYNAPSE models.

Provides the ``train()`` entry point that builds a Lightning Trainer
with all best practices (AMP, gradient clipping, EMA, checkpointing,
W&B logging) and runs ``Trainer.fit()``.

This replaces the previous manual training loop with a production-grade
Lightning pipeline that mirrors the baselines training engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from synapse.common.config import SimpleNamespaceConfig, load_config
from synapse.arch.losses.combined_loss import LossConfig

from ..builders.builder import build_model_from_cfg, resolve_normalization
from ..evaluation.hooks import run_post_training_evaluation
from .data_module import SynapseDataModule
from .module import SynapseLightningModule
from ..utils.callbacks import EMACallback, PostTrainEvalCallback

log = logging.getLogger(__name__)


def _normalize_config(config: str | Path | dict | DictConfig) -> Any:
    """Normalize config to SimpleNamespaceConfig.

    Accepts:
    - str/Path: path to YAML file (legacy)
    - dict/DictConfig: already loaded config (Hydra)
    """
    if isinstance(config, (str, Path)):
        return load_config(config)
    elif isinstance(config, DictConfig):
        # Convert DictConfig to dict, then to SimpleNamespaceConfig
        return SimpleNamespaceConfig(OmegaConf.to_container(config, resolve=True))
    elif isinstance(config, dict):
        return SimpleNamespaceConfig(config)
    else:
        raise TypeError(f"Invalid config type: {type(config)}")


def train(
    config: str | Path | dict | DictConfig,
    output_dir: str,
    *,
    dataset_name: str | None = None,
    max_epochs: int | None = None,
    accelerator: str | None = None,
    devices: int | list[int] | str | None = None,
    precision: str | int | None = None,
    warmup_steps: int = 500,
    gradient_clip_norm: float = 1.0,
    ema_decay: float = 0.999,
    early_stopping_patience: int | None = 10,
    save_top_k: int = 1,
    log_every_n_steps: int = 10,
    enable_checkpointing: bool = True,
    enable_progress_bar: bool = True,
    logger_name: str | None = "wandb",
) -> dict[str, Any]:
    """Train a Z3 SYNAPSE model end-to-end using PyTorch Lightning.

    Parameters
    ----------
    config : str, Path, dict, or DictConfig
        Configuration. Can be a path to YAML file (legacy) or a
        pre-loaded config dict/DictConfig (Hydra).
    output_dir : str
        Output directory for checkpoints, logs, and evaluation.
    dataset_name : str or None
        Dataset name from the adapter registry.
    max_epochs : int or None
        Override max epochs.  If None, reads from config.
    accelerator : str or None
        Lightning accelerator (``"gpu"``, ``"cpu"``, ``"auto"``).
    devices : int, list[int], str, or None
        Number of devices or specific device IDs.
    precision : str, int, or None
        Training precision (``"bf16-mixed"``, ``"16-mixed"``, ``32``).
    warmup_steps : int
        Number of linear warmup steps for LR schedule.
    gradient_clip_norm : float
        Max gradient norm for clipping.
    ema_decay : float
        EMA decay factor (0.999 = slow, 0.9 = fast).
    early_stopping_patience : int or None
        Patience for early stopping on val/loss.  None to disable.
    save_top_k : int
        Number of best checkpoints to keep.
    log_every_n_steps : int
        Logging frequency.
    enable_checkpointing : bool
        Whether to save checkpoints.
    enable_progress_bar : bool
        Whether to show progress bar.
    logger_name : str or None
        Logger name (``"wandb"``, ``"tensorboard"``, None to disable).

    Returns
    -------
    dict with keys "history", "test_metrics", "bundle", "evaluation".
    """
    cfg = _normalize_config(config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data module
    # ------------------------------------------------------------------
    datamodule = SynapseDataModule(cfg, dataset_name=dataset_name)
    datamodule.setup()

    # ------------------------------------------------------------------
    # 2. Model + normalization
    # ------------------------------------------------------------------
    model = build_model_from_cfg(cfg, datamodule.bundle)
    norm = resolve_normalization(datamodule.bundle)
    model.set_normalization(
        torch.from_numpy(norm["mu"]).float(),
        torch.from_numpy(norm["sigma"]).float(),
    )

    # ------------------------------------------------------------------
    # 3. LightningModule
    # ------------------------------------------------------------------
    loss_config = LossConfig()
    lightning_model = SynapseLightningModule(
        model=model,
        config=model.config,
        loss_config=loss_config,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
        gradient_clip_norm=gradient_clip_norm,
    )

    # ------------------------------------------------------------------
    # 4. Callbacks
    # ------------------------------------------------------------------
    callbacks: list[pl.Callback] = []

    # EMA shadow model
    callbacks.append(EMACallback(decay=ema_decay))

    # Post-training evaluation
    ds_name = dataset_name or getattr(
        getattr(cfg, "data", None), "dataset", "synthetic",
    )
    callbacks.append(PostTrainEvalCallback(
        dataset_name=ds_name,
        output_root=output_root,
    ))

    # Early stopping
    if early_stopping_patience is not None:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        ))

    # Checkpointing
    if enable_checkpointing:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=output_root / "checkpoints",
            filename="best-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
        ))

    # ------------------------------------------------------------------
    # 5. Logger
    # ------------------------------------------------------------------
    trainer_logger = None
    if logger_name == "wandb":
        try:
            trainer_logger = pl.loggers.WandbLogger(
                project="synapse-z3",
                name=ds_name,
                save_dir=str(output_root / "wandb"),
            )
        except ImportError:
            log.warning("wandb not installed — falling back to CSV logger")
            trainer_logger = pl.loggers.CSVLogger(save_dir=str(output_root / "logs"))
    elif logger_name == "tensorboard":
        trainer_logger = pl.loggers.TensorBoardLogger(save_dir=str(output_root / "logs"))
    elif logger_name is not None:
        trainer_logger = pl.loggers.CSVLogger(save_dir=str(output_root / "logs"))

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    epochs = max_epochs or cfg.training.epochs

    # Auto-detect accelerator
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Auto-detect precision
    if precision is None:
        if accelerator == "gpu" and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        elif accelerator == "gpu":
            precision = "16-mixed"
        else:
            precision = 32

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices or ("auto" if accelerator == "gpu" else 1),
        precision=precision,
        callbacks=callbacks,
        logger=trainer_logger,
        default_root_dir=str(output_root),
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=gradient_clip_norm,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=1,
        deterministic=False,
        benchmark=True,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    log.info(
        "Starting Lightning training: epochs=%d, accelerator=%s, precision=%s",
        epochs, accelerator, precision,
    )
    trainer.fit(lightning_model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # 8. Test
    # ------------------------------------------------------------------
    test_results = trainer.test(lightning_model, datamodule=datamodule, verbose=True)

    # ------------------------------------------------------------------
    # 9. Collect results (backward-compatible format)
    # ------------------------------------------------------------------
    history = _extract_history(trainer)

    test_metrics = {}
    if test_results:
        test_metrics = {
            "loss": test_results[0].get("test/loss", float("nan")),
            "accuracy": test_results[0].get("test/accuracy", float("nan")),
        }

    # Post-training evaluation is handled by PostTrainEvalCallback
    eval_results = {}

    return {
        "history": history,
        "test_metrics": test_metrics,
        "bundle": datamodule.bundle,
        "evaluation": eval_results,
        "trainer": trainer,
        "lightning_module": lightning_model,
    }


def _extract_history(trainer: pl.Trainer) -> list[dict[str, Any]]:
    """Extract epoch-level training history from the Lightning Trainer."""
    history: list[dict[str, Any]] = []

    # Try to get metrics from the CSV logger
    csv_logger = None
    for lg in trainer.loggers:
        if isinstance(lg, pl.loggers.CSVLogger):
            csv_logger = lg
            break

    if csv_logger is not None:
        metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
        if metrics_path.exists():
            import pandas as pd
            df = pd.read_csv(metrics_path)
            for epoch in df.get("epoch", []).unique():
                epoch_data = df[df["epoch"] == epoch]
                row: dict[str, Any] = {"epoch": int(epoch)}
                for col in epoch_data.columns:
                    if col in ("epoch", "step") or "/" not in col:
                        continue
                    val = epoch_data[col].dropna().mean()
                    if not np.isnan(val):
                        row[col] = float(val)
                history.append(row)

    return history
