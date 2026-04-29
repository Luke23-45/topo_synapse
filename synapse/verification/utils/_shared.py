from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from synapse.common.generators import piecewise_constant, piecewise_constant_auto, random_walk
from synapse.empirical.datasets.synthetic_topology import build_synthetic_bundle
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.random import seed_everything

from synapse.verification.config import get_experiment_config
from synapse.verification.utils.artifact_writer import ArtifactWriter


class _Grid:
    def __init__(self, axes: dict[str, Sequence[Any]]) -> None:
        self.names = tuple(axes)
        self.values = tuple(axes[name] for name in self.names)
        self.size = 1
        for vals in self.values:
            self.size *= len(vals)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for combo in product(*self.values):
            yield dict(zip(self.names, combo))


def iter_parameter_grid(**axes: Sequence[Any]) -> _Grid:
    return _Grid(axes)


def make_config(
    *,
    input_dim: int = 2,
    output_dim: int = 4,
    K: int = 8,
    r: int = 1,
    Q: int = 1,
    k: int = 16,
    max_history_tokens: int = 64,
) -> SynapseConfig:
    return SynapseConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=48,
        d_model=48,
        num_heads=4,
        num_layers=1,
        ffn_ratio=2,
        dropout=0.0,
        K=K,
        r=r,
        lam=0.5,
        Q=Q,
        k=k,
        max_history_tokens=max_history_tokens,
        batch_size=16,
        epochs=1,
        train_size=64,
        val_size=32,
        test_size=32,
        noise_std=0.03,
        seed=7,
    )


def make_config_from_yaml(
    experiment_id: str,
    *,
    K_override: int | None = None,
    r_override: int | None = None,
    max_history_tokens_override: int | None = None,
) -> SynapseConfig:
    """Build a SynapseConfig from the model block of verification.yaml.

    Optional overrides let grid-search scripts (e.g. VZ3-04) vary
    specific model parameters per iteration while keeping the rest
    from the YAML.
    """
    cfg = get_experiment_config(experiment_id)
    model_cfg = cfg.get("model", {})
    return make_config(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 4),
        K=K_override if K_override is not None else model_cfg.get("K", 8),
        r=r_override if r_override is not None else model_cfg.get("r", 1),
        Q=model_cfg.get("Q", 1),
        k=model_cfg.get("k", 16),
        max_history_tokens=max_history_tokens_override if max_history_tokens_override is not None else model_cfg.get("max_history_tokens", 64),
    )


def build_model(config: SynapseConfig) -> Z3TopologyFirstModel:
    seed_everything(config.seed)
    bundle = build_synthetic_bundle(
        config.train_size,
        config.val_size,
        config.test_size,
        length=config.max_history_tokens,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    model = Z3TopologyFirstModel(config)
    model.refresh_normalization(bundle.train_sequences)
    model.eval()
    return model


def run_standalone(
    *,
    experiment_id: str,
    run_experiment_fn,
) -> int:
    """Standard CLI entry-point for a verification script.

    The *run_experiment_fn* signature is:
        (config: dict, writer: ArtifactWriter, verbose: bool) -> dict
    It returns a summary dict that is persisted as ``summary.json``.
    """
    parser = argparse.ArgumentParser(description=f"Verification: {experiment_id}")
    parser.add_argument("--output-dir", default="verification_outputs")
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
    "build_model",
    "get_experiment_config",
    "iter_parameter_grid",
    "make_config",
    "make_config_from_yaml",
    "piecewise_constant",
    "piecewise_constant_auto",
    "random_walk",
    "run_standalone",
]
