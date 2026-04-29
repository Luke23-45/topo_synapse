"""Topology-specific visualization for Z3 SYNAPSE evaluation.

Generates figures for:
    - Persistence diagrams (birth vs death scatter)
    - Point cloud visualizations (PCA-projected)
    - Proxy-exact alignment scatter plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not available — topology visualization will be skipped")

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def plot_point_cloud(
    cloud: np.ndarray,
    path: str | Path,
    anchor_times: np.ndarray | None = None,
) -> Optional[Path]:
    """Plot a topology lift point cloud (2D or PCA-projected).

    Parameters
    ----------
    cloud : np.ndarray, shape (N, D)
        Point cloud coordinates.
    path : str or Path
        Output file path.
    anchor_times : np.ndarray, shape (N,), optional
        Normalized time values for coloring.
    """
    if not HAS_MATPLOTLIB:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if cloud.size == 0:
        return None

    # Reduce to 2D if needed
    if cloud.shape[1] > 2 and HAS_SKLEARN:
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(cloud)
    elif cloud.shape[1] > 2:
        points_2d = cloud[:, :2]
    else:
        points_2d = cloud[:, :2]

    fig, ax = plt.subplots(figsize=(6, 6))
    if anchor_times is not None:
        scatter = ax.scatter(
            points_2d[:, 0], points_2d[:, 1],
            c=anchor_times, cmap="viridis", s=20,
        )
        fig.colorbar(scatter, label="Normalized Time")
    else:
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=20, alpha=0.7)

    ax.set_title("Topology Lift Cloud", fontsize=12, fontweight="bold")
    ax.set_xlabel("dim0", fontsize=10)
    ax.set_ylabel("dim1", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved point cloud to %s", path)
    return path


def plot_persistence_diagram(
    diagrams: list[list[tuple[float, float]]],
    output_path: str | Path,
) -> Optional[Path]:
    """Plot persistence diagrams for all homology degrees.

    Parameters
    ----------
    diagrams : list of list of (birth, death)
        Persistence diagrams per homology degree.
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for q, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        dgm = np.array(dgm)
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        finite_deaths = deaths[np.isfinite(deaths)]
        max_death = np.max(finite_deaths) if len(finite_deaths) > 0 else np.max(births) + 1.0
        inf_replacement = max_death * 1.1
        plot_deaths = np.where(np.isfinite(deaths), deaths, inf_replacement)
        ax.scatter(
            births, plot_deaths,
            color=colors[q % len(colors)],
            label=f"H{q}", alpha=0.7, s=20,
        )

    # Diagonal reference
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k-", alpha=0.3, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Persistence Diagram", fontsize=12, fontweight="bold")
    ax.set_xlabel("Birth", fontsize=10)
    ax.set_ylabel("Death", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved persistence diagram to %s", output_path)
    return output_path


def plot_proxy_exact_scatter(
    proxy_features: np.ndarray,
    exact_summaries: np.ndarray,
    output_path: str | Path,
) -> Optional[Path]:
    """Scatter plot of proxy vs exact topology features.

    Parameters
    ----------
    proxy_features : np.ndarray, shape (B, D_proxy)
    exact_summaries : np.ndarray, shape (B, D_exact)
    output_path : str or Path
    """
    if not HAS_MATPLOTLIB:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proxy = proxy_features.reshape(proxy_features.shape[0], -1)
    exact = exact_summaries.reshape(exact_summaries.shape[0], -1)

    # Use first principal component of each for 1D scatter
    if proxy.shape[1] > 1 and HAS_SKLEARN:
        proxy_1d = PCA(n_components=1).fit_transform(proxy)[:, 0]
        exact_1d = PCA(n_components=1).fit_transform(exact)[:, 0]
    else:
        proxy_1d = proxy[:, 0]
        exact_1d = exact[:, 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(proxy_1d, exact_1d, alpha=0.5, s=20, color="#3498db")

    # Diagonal
    max_val = max(proxy_1d.max(), exact_1d.max())
    min_val = min(proxy_1d.min(), exact_1d.min())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.3, label="Equal")

    ax.set_title("Proxy vs Exact Topology", fontsize=12, fontweight="bold")
    ax.set_xlabel("Proxy (1st PC)", fontsize=10)
    ax.set_ylabel("Exact (1st PC)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved proxy-exact scatter to %s", output_path)
    return output_path
