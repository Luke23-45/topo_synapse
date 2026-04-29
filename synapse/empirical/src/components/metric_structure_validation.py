from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config, exact_summary_matrix


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    n_samples = p["n_samples"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    x = exact_summary_matrix(model, bundle.test_sequences[:n_samples])
    y = bundle.test_labels[:n_samples]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    same = d[y[:, None] == y[None, :]]
    diff = d[y[:, None] != y[None, :]]
    mean_same = float(np.mean(same))
    mean_diff = float(np.mean(diff))
    passed = mean_same < mean_diff

    row = {
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "class_distance_gap": passed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("distance_matrix.npy", d)
    writer.save_numpy("summary_features.npy", x)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "mean_same": mean_same,
        "mean_diff": mean_diff,
        "class_distance_gap": passed,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-04",
            run_experiment_fn=run_experiment,
        )
    )
