from __future__ import annotations

import argparse

import numpy as np

from synapse.common.runtime import load_config
from synapse.empirical.datasets.synthetic_topology import build_synthetic_bundle
from synapse.utils.io import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare synthetic topology datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="synapse_outputs/dataset")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = build_synthetic_bundle(
        config.train_size,
        config.val_size,
        config.test_size,
        length=config.max_history_tokens,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    output_dir = ensure_dir(args.output_dir)
    np.savez(
        output_dir / "synthetic_topology.npz",
        train_sequences=bundle.train_sequences,
        train_labels=bundle.train_labels,
        val_sequences=bundle.val_sequences,
        val_labels=bundle.val_labels,
        test_sequences=bundle.test_sequences,
        test_labels=bundle.test_labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
