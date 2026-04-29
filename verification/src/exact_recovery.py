from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.verification.utils._shared import ArtifactWriter, build_model, make_config_from_yaml, piecewise_constant


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    num_segments = params_cfg["num_segments"]
    seed = params_cfg["seed"]
    tolerance = params_cfg["tolerance"]
    input_dim = params_cfg["input_dim"]
    max_history_tokens = params_cfg["max_history_tokens"]

    t0 = perf_counter()

    model_cfg = make_config_from_yaml(config["experiment_id"])
    model = build_model(model_cfg)
    seq = piecewise_constant(input_dim, max_history_tokens, num_segments=num_segments, seed=seed)

    audit_a = model.exact_audit(torch.from_numpy(seq).float().unsqueeze(0))[0]
    audit_b = model.exact_audit(torch.from_numpy(seq).float().unsqueeze(0))[0]

    indices_match = audit_a.anchor_indices == audit_b.anchor_indices
    shapes_match = audit_a.point_cloud.shape == audit_b.point_cloud.shape
    cloud_close = True
    max_abs_diff = 0.0
    if audit_a.point_cloud.size:
        diff = np.abs(audit_a.point_cloud - audit_b.point_cloud)
        max_abs_diff = float(diff.max())
        cloud_close = bool((diff < tolerance).all())

    deterministic = indices_match and shapes_match and cloud_close

    row = {
        "tolerance": tolerance,
        "indices_match": indices_match,
        "shapes_match": shapes_match,
        "cloud_close": cloud_close,
        "max_abs_diff": max_abs_diff,
        "deterministic": deterministic,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    if audit_a.point_cloud.size:
        writer.save_numpy("cloud_a.npy", audit_a.point_cloud)
        writer.save_numpy("cloud_b.npy", audit_b.point_cloud)

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
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-06",
            run_experiment_fn=run_experiment,
        )
    )
