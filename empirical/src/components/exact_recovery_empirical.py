from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    seq = torch.from_numpy(bundle.test_sequences[0]).float().unsqueeze(0)
    a = model.exact_audit(seq)[0]
    b = model.exact_audit(seq)[0]

    indices_match = a.anchor_indices == b.anchor_indices
    shapes_match = a.point_cloud.shape == b.point_cloud.shape
    cloud_close = True
    max_abs_diff = 0.0
    if a.point_cloud.size:
        diff = np.abs(a.point_cloud - b.point_cloud)
        max_abs_diff = float(diff.max())
        cloud_close = bool((diff < 1e-8).all())

    deterministic = indices_match and shapes_match and cloud_close

    row = {
        "anchor_count": len(a.anchor_indices),
        "indices_match": indices_match,
        "shapes_match": shapes_match,
        "cloud_close": cloud_close,
        "max_abs_diff": max_abs_diff,
        "deterministic": deterministic,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    if a.point_cloud.size:
        writer.save_numpy("cloud_a.npy", a.point_cloud)
        writer.save_numpy("cloud_b.npy", b.point_cloud)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "deterministic": deterministic,
        "max_abs_diff": max_abs_diff,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-07",
            run_experiment_fn=run_experiment,
        )
    )
