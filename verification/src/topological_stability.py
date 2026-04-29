from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.synapse_core.topology import hausdorff_distance
from synapse.verification.utils._shared import ArtifactWriter, build_model, make_config_from_yaml, random_walk


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    seed = params_cfg["seed"]
    noise_seed = params_cfg["noise_seed"]
    noise_std = params_cfg["noise_std"]
    input_dim = params_cfg["input_dim"]
    max_history_tokens = params_cfg["max_history_tokens"]

    t0 = perf_counter()

    model_cfg = make_config_from_yaml(config["experiment_id"])
    model = build_model(model_cfg)

    clean = random_walk(input_dim, max_history_tokens, seed=seed)
    noisy = clean + np.random.default_rng(noise_seed).normal(scale=noise_std, size=clean.shape)

    audit_a = model.exact_audit(torch.from_numpy(clean).float().unsqueeze(0))[0]
    audit_b = model.exact_audit(torch.from_numpy(noisy).float().unsqueeze(0))[0]

    hd = hausdorff_distance(audit_a.point_cloud, audit_b.point_cloud)
    finite = bool(np.isfinite(hd))

    row = {
        "noise_std": noise_std,
        "hausdorff_distance": float(hd),
        "finite": finite,
        "anchor_count_clean": len(audit_a.anchor_indices),
        "anchor_count_noisy": len(audit_b.anchor_indices),
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    if audit_a.point_cloud.size:
        writer.save_numpy("cloud_clean.npy", audit_a.point_cloud)
    if audit_b.point_cloud.size:
        writer.save_numpy("cloud_noisy.npy", audit_b.point_cloud)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "hausdorff_distance": float(hd),
        "finite": finite,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-07",
            run_experiment_fn=run_experiment,
        )
    )
