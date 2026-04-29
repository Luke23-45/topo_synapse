"""CLI script for visualizing Z3 SYNAPSE topology outputs.

Generates point cloud plots, persistence diagrams, and proxy-exact
alignment scatter plots from a trained model checkpoint.

Usage:
    python -m synapse.evaluation.scripts.visualize \\
        --checkpoint path/to/best.pt \\
        --output-dir synapse_outputs/vis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from synapse.common.runtime import load_config
from synapse.evaluation.visualization.topology_plots import (
    plot_persistence_diagram,
    plot_point_cloud,
    plot_proxy_exact_scatter,
)
from synapse.synapse_arch.model import Z3TopologyFirstModel

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Z3 topology outputs.")
    parser.add_argument("--config", default=None, help="Path to model config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="synapse_outputs/vis", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Z3TopologyFirstModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic sequences for visualization
    from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset
    sequences, labels, names = generate_topology_dataset(
        args.num_samples,
        length=config.max_history_tokens,
        noise_std=config.noise_std,
        seed=config.seed + 40,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    seq_tensor = torch.from_numpy(sequences).float().to(device)

    with torch.no_grad():
        out = model(seq_tensor)

    audits = model.exact_audit(seq_tensor.cpu())

    for i in range(args.num_samples):
        sample_dir = output_dir / f"sample_{i:02d}"
        sample_dir.mkdir(exist_ok=True)

        audit = audits[i]

        # Point cloud
        anchor_times = np.array(audit.anchor_indices) / max(len(audit.anchor_indices), 1)
        plot_point_cloud(audit.point_cloud, sample_dir / "point_cloud.png", anchor_times=anchor_times)

        # Persistence diagram
        plot_persistence_diagram(audit.persistence_diagrams, sample_dir / "persistence_diagram.png")

        log.info("Saved visualizations for sample %d to %s", i, sample_dir)

    # Proxy-exact alignment scatter (all samples)
    exact_summaries = np.stack([a.topology_summary for a in audits], axis=0)
    plot_proxy_exact_scatter(
        out.proxy_features.detach().cpu().numpy(),
        exact_summaries,
        output_dir / "proxy_exact_scatter.png",
    )

    log.info("Visualization complete. Outputs saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
