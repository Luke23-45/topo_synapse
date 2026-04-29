from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from synapse.empirical.config import get_experiment_config
from synapse.empirical.datasets.synthetic_topology import build_synthetic_bundle
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.random import seed_everything
from synapse.verification.utils.artifact_writer import ArtifactWriter


def build_empirical_model(seed: int, length: int, noise_std: float, train_size: int = 128, val_size: int = 64, test_size: int = 64) -> tuple[Z3TopologyFirstModel, Any]:
    seed_everything(seed)
    bundle = build_synthetic_bundle(train_size, val_size, test_size, length=length, noise_std=noise_std, seed=seed)
    config = SynapseConfig(
        input_dim=2,
        output_dim=4,
        hidden_dim=48,
        d_model=48,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2,
        dropout=0.0,
        K=8,
        r=1,
        lam=0.5,
        Q=1,
        k=16,
        max_history_tokens=length,
        batch_size=32,
        epochs=2,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        noise_std=noise_std,
        seed=seed,
    )
    model = Z3TopologyFirstModel(config)
    model.refresh_normalization(bundle.train_sequences)
    model.eval()
    return model, bundle


def build_empirical_model_from_config(config: dict[str, Any]) -> tuple[Z3TopologyFirstModel, Any]:
    """Build model + bundle from an empirical YAML config dict."""
    p = config["params"]
    model_cfg = config.get("model", {})
    seed = p["seed"]
    length = p["length"]
    noise_std = p["noise_std"]
    train_size = p.get("train_size", model_cfg.get("train_size", 128))
    val_size = p.get("val_size", model_cfg.get("val_size", 64))
    test_size = p.get("test_size", model_cfg.get("test_size", 64))

    seed_everything(seed)
    bundle = build_synthetic_bundle(train_size, val_size, test_size, length=length, noise_std=noise_std, seed=seed)
    synapse_cfg = SynapseConfig(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 4),
        hidden_dim=48,
        d_model=48,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2,
        dropout=0.0,
        K=model_cfg.get("K", 8),
        r=model_cfg.get("r", 1),
        lam=0.5,
        Q=model_cfg.get("Q", 1),
        k=model_cfg.get("k", 16),
        max_history_tokens=model_cfg.get("max_history_tokens", length),
        batch_size=32,
        epochs=2,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        noise_std=noise_std,
        seed=seed,
    )
    model = Z3TopologyFirstModel(synapse_cfg)
    model.refresh_normalization(bundle.train_sequences)
    model.eval()
    return model, bundle


def exact_summary_matrix(model: Z3TopologyFirstModel, sequences: np.ndarray) -> np.ndarray:
    audits = model.exact_audit(torch.from_numpy(sequences).float())
    return np.stack([audit.topology_summary for audit in audits], axis=0)


def run_standalone(
    *,
    experiment_id: str,
    run_experiment_fn,
) -> int:
    """Standard CLI entry-point for an empirical experiment script.

    The *run_experiment_fn* signature is:
        (config: dict, writer: ArtifactWriter, verbose: bool) -> dict
    It returns a summary dict that is persisted as ``summary.json``.
    """
    parser = argparse.ArgumentParser(description=f"Empirical: {experiment_id}")
    parser.add_argument("--output-dir", default="empirical_outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = get_experiment_config(experiment_id)
    writer = ArtifactWriter(args.output_dir, experiment_id)

    summary = run_experiment_fn(config=config, writer=writer, verbose=args.verbose)

    writer.save_json("summary.json", summary)
    print(f"[{experiment_id}] Done. Output -> {writer.root}")
    return 0


__all__ = [
    "ArtifactWriter",
    "build_empirical_model",
    "build_empirical_model_from_config",
    "exact_summary_matrix",
    "get_experiment_config",
    "run_standalone",
]
