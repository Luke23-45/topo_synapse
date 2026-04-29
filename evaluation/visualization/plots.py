"""Standard metric plots for Z3 SYNAPSE evaluation.

Provides publication-quality matplotlib figures for:
    - Confusion matrices
    - Learning curves (train/val loss over epochs)
    - Accuracy comparison bar charts
    - Robustness sweep curves (noise, rotation, length)
    - Feature importance bar charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not available — visualization will be skipped")


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

FIG_DPI = 300
FIG_WIDTH = 8
FIG_HEIGHT = 5

PALETTE = [
    "#e74c3c",  # red
    "#3498db",  # blue
    "#2ecc71",  # green
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#34495e",  # dark gray
]


def _setup_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    output_path: str | Path = "confusion_matrix.pdf",
) -> Optional[Path]:
    """Plot a confusion matrix as a heatmap."""
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"C{i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.8), max(5, n_classes * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=text_color, fontsize=9)

    _setup_axes(ax, "Confusion Matrix", "Predicted", "True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix to %s", output_path)
    return output_path


def plot_learning_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    output_path: str | Path = "learning_curves.pdf",
) -> Optional[Path]:
    """Plot train/val loss over epochs for one or more conditions."""
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 2, FIG_HEIGHT))

    for idx, (cond_key, losses) in enumerate(train_losses.items()):
        color = PALETTE[idx % len(PALETTE)]
        ax1.plot(losses, color=color, label=cond_key, alpha=0.8)
    _setup_axes(ax1, "Training Loss", "Epoch", "Loss")
    ax1.legend(fontsize=8)

    for idx, (cond_key, losses) in enumerate(val_losses.items()):
        color = PALETTE[idx % len(PALETTE)]
        ax2.plot(losses, color=color, label=cond_key, alpha=0.8)
    _setup_axes(ax2, "Validation Loss", "Epoch", "Loss")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved learning curves to %s", output_path)
    return output_path


def plot_accuracy_comparison(
    condition_means: Dict[str, float],
    condition_stds: Dict[str, float],
    metric_name: str = "Accuracy",
    output_path: str | Path = "accuracy_comparison.pdf",
) -> Optional[Path]:
    """Grouped bar chart comparing a metric across conditions."""
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    conditions = list(condition_means.keys())
    means = [condition_means[c] for c in conditions]
    stds = [condition_stds[c] for c in conditions]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(conditions))]

    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                  capsize=5, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha="right")
    _setup_axes(ax, f"{metric_name} Comparison", "Condition", metric_name)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved accuracy comparison to %s", output_path)
    return output_path


def plot_robustness_sweep(
    sweep_data: list[dict[str, float]],
    x_key: str,
    y_key: str = "accuracy",
    title: str = "Robustness Sweep",
    output_path: str | Path = "robustness_sweep.pdf",
) -> Optional[Path]:
    """Plot a robustness sweep curve (e.g. accuracy vs noise level)."""
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    x_vals = [d[x_key] for d in sweep_data]
    y_vals = [d[y_key] for d in sweep_data]

    ax.plot(x_vals, y_vals, color=PALETTE[0], linewidth=2, marker="o", markersize=6)
    ax.fill_between(
        x_vals,
        [v * 0.95 for v in y_vals],
        [min(v * 1.05, 1.0) for v in y_vals],
        color=PALETTE[0],
        alpha=0.15,
    )

    _setup_axes(ax, title, x_key, y_key)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved robustness sweep to %s", output_path)
    return output_path


def plot_feature_importance(
    importance_scores: list[float],
    output_path: str | Path = "feature_importance.pdf",
    top_k: int | None = None,
) -> Optional[Path]:
    """Bar chart of per-dimension feature importance."""
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = np.array(importance_scores)
    if top_k is not None:
        top_indices = np.argsort(scores)[-top_k:]
        scores = scores[top_indices]
        labels = [f"dim_{i}" for i in top_indices]
    else:
        labels = [f"dim_{i}" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(FIG_HEIGHT * 0.5, len(scores) * 0.3)))
    y_pos = np.arange(len(scores))
    ax.barh(y_pos, scores, color=PALETTE[0], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    _setup_axes(ax, "Feature Importance (Confidence Drop)", "Importance", "Dimension")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved feature importance to %s", output_path)
    return output_path
