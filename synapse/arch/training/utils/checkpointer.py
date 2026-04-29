"""Checkpoint utilities for Z3 SYNAPSE training.

When using the Lightning pipeline, checkpointing is handled by
``ModelCheckpoint`` callback.  This module provides lightweight
save/load helpers for standalone scripts and backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int = 0,
    config: dict | None = None,
    *,
    scheduler=None,
    ema_state: dict | None = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : str or Path
        Output file path.
    model : nn.Module or LightningModule
        Model to save.  If LightningModule, the underlying ``.model``
        is extracted automatically.
    optimizer : Optimizer or None
    epoch : int
    config : dict or None
    scheduler : LRScheduler or None
    ema_state : dict or None
        EMA shadow weights (if applicable).
    """
    base_model = getattr(model, "model", model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": base_model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        state["config"] = config
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if ema_state is not None:
        state["ema_state_dict"] = ema_state

    torch.save(state, path)
    log.debug("Checkpoint saved to %s", path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    scheduler=None,
    device: torch.device | str | None = None,
) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    path : str or Path
        Checkpoint file path.
    model : nn.Module or LightningModule
        Model to load weights into.
    optimizer : Optimizer or None
    scheduler : LRScheduler or None
    device : torch.device, str, or None
        Map location for loading.

    Returns
    -------
    dict
        The full checkpoint dictionary (includes epoch, config, etc.).
    """
    base_model = getattr(model, "model", model)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or ("cpu" if not torch.cuda.is_available() else None)
    checkpoint = torch.load(path, weights_only=False, map_location=map_location)

    base_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    log.info("Checkpoint loaded from %s (epoch %d)", path, checkpoint.get("epoch", -1))
    return checkpoint
