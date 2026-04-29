from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.verification.utils._shared import ArtifactWriter, build_model, make_config_from_yaml, random_walk


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    seed = params_cfg["seed"]
    input_dim = params_cfg["input_dim"]
    max_history_tokens = params_cfg["max_history_tokens"]

    t0 = perf_counter()

    model_cfg = make_config_from_yaml(config["experiment_id"])
    model = build_model(model_cfg)
    seq = random_walk(input_dim, max_history_tokens, seed=seed)
    audit = model.exact_audit(torch.from_numpy(seq).float().unsqueeze(0))[0]

    projected = audit.anchor_vectors
    cloud_dim_ok = audit.point_cloud.shape[1] == model_cfg.k if audit.point_cloud.size else True
    time_zeroed = bool(np.allclose(projected[:, 0], 0.0)) if projected.size else True

    row = {
        "k": model_cfg.k,
        "cloud_shape": list(audit.point_cloud.shape),
        "cloud_dim_ok": cloud_dim_ok,
        "time_zeroed": time_zeroed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    if audit.point_cloud.size:
        writer.save_numpy("point_cloud.npy", audit.point_cloud)
    if projected.size:
        writer.save_numpy("anchor_vectors.npy", projected)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "cloud_dim_ok": cloud_dim_ok,
        "time_zeroed": time_zeroed,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-05",
            run_experiment_fn=run_experiment,
        )
    )
