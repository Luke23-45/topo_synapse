"""CLI script for running Z3 SYNAPSE ablation experiments.

Trains each ablation variant for a few epochs, evaluates, and
compiles results into a JSON report.

Usage:
    python -m synapse.evaluation.scripts.run_ablations \\
        --config config/default.yaml \\
        --output synapse_outputs/ablations.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from synapse.common.runtime import load_config
from synapse.evaluation.ablations import build_ablation_configs
from synapse.evaluation.runners.classification import ClassificationEvaluator
from synapse.evaluation.reporting import generate_json_report, generate_markdown_report
from synapse.arch.data.data import build_dataloaders
from synapse.arch.losses.combined_loss import LossConfig, compute_loss
from synapse.arch.training.utils.checkpointer import save_checkpoint
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.io import ensure_dir, save_json
from synapse.utils.random import seed_everything

log = logging.getLogger(__name__)


def _train_ablation(config, output_dir: Path, dataset_name: str = "synthetic") -> dict:
    """Train a single ablation variant and return evaluation results."""
    seed_everything(config.seed)

    _, _, test_loader, bundle = build_dataloaders(config, dataset_name=dataset_name)
    model = Z3TopologyFirstModel(config)
    model.refresh_normalization(bundle.train_sequences)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_cfg = LossConfig()

    # Build train loader from bundle
    from synapse.synapse.data.trajectory_dataset import TrajectoryDataset
    from synapse.synapse.data.collate import trajectory_collate_fn
    train_loader = DataLoader(
        TrajectoryDataset(bundle.train_sequences, bundle.train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate_fn,
    )

    # Short training run for ablation
    ablation_epochs = max(1, min(3, config.epochs))
    for epoch in range(ablation_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(batch["sequences"])
            loss, _ = compute_loss(out, batch, loss_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    evaluator = ClassificationEvaluator(config, test_loader, output_dir)
    result = evaluator.evaluate(model)
    return result.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Z3 ablation experiments.")
    parser.add_argument("--config", default=None, help="Path to model config YAML")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name")
    parser.add_argument("--output", default="synapse_outputs/ablations.json")
    args = parser.parse_args()

    base = load_config(args.config)
    output_dir = ensure_dir(Path(args.output).parent)

    results = {}
    for name, ablated in build_ablation_configs(base).items():
        log.info("Running ablation: %s", name)
        results[name] = _train_ablation(ablated, output_dir, dataset_name=args.dataset)

    save_json(args.output, results)
    log.info("Ablation results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
