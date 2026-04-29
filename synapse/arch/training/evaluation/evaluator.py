"""Simple validation/test evaluation for Z3 SYNAPSE models.

Works with both plain ``nn.Module`` and ``LightningModule`` models.
When using the Lightning pipeline, prefer ``Trainer.test()`` which
handles EMA swapping automatically.  This function is kept for
standalone evaluation and backward compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def evaluate(model, loader, loss_fn, device=None):
    """Evaluate a model on a dataloader.

    Parameters
    ----------
    model : nn.Module or LightningModule
        The model to evaluate.  If a LightningModule, the underlying
        ``.model`` attribute is used for the forward pass.
    loader : DataLoader
    loss_fn : callable
        Loss function ``fn(output, batch) -> (loss, parts)``.
    device : torch.device or None
        Device to use.  If None, auto-detects from model parameters.

    Returns
    -------
    dict with keys "loss", "accuracy".
    """
    # Unwrap LightningModule if needed
    base_model = getattr(model, "model", model)
    base_model.eval()

    if device is None:
        device = next(base_model.parameters()).device

    losses = []
    accuracies = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = base_model(batch["sequences"])
            loss, _ = loss_fn(out, batch)
            losses.append(float(loss.item()))
            pred = out.logits.argmax(dim=-1)
            accuracies.append(float((pred == batch["targets"]).float().mean().item()))

    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}
