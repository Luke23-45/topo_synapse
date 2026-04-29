"""Visualization — Z3 Baseline Study.

Generates publication-quality plots for the baseline comparison.
Mirrors the legacy ``baselines/src/reporting/visualize.py`` pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    log.warning("matplotlib not available — plots will be skipped")


def plot_accuracy_comparison(
    accuracies: Dict[str, float],
    stds: Dict[str, float] | None = None,
    output_path: Path | str = "accuracy_comparison.pdf",
) -> None:
    """Bar chart comparing backbone accuracies."""
    if not HAS_MPL:
        return

    names = list(accuracies.keys())
    vals = [accuracies[n] for n in names]
    errs = [stds[n] for n in names] if stds else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals, yerr=errs, capsize=5, color="steelblue", edgecolor="black")
    ax.set_ylabel("Accuracy")
    ax.set_title("Z3 Baseline Study — Classification Accuracy")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    log.info("Saved accuracy plot to %s", output_path)


def plot_learning_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    output_path: Path | str = "learning_curves.pdf",
) -> None:
    """Line plot of training and validation loss curves."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in train_losses.items():
        ax.plot(losses, label=f"{name} (train)", linestyle="--")
    for name, losses in val_losses.items():
        ax.plot(losses, label=f"{name} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Z3 Baseline Study — Learning Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path))
    plt.close(fig)
    log.info("Saved learning curves to %s", output_path)


__all__ = ["plot_accuracy_comparison", "plot_learning_curves"]
