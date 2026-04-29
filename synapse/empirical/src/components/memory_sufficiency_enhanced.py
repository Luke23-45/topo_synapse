from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.empirical.common.math_utils import ridge_regression_mse
from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config, exact_summary_matrix
from synapse.empirical.common.tasks import generate_memory_task_dataset


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    n_samples = p["n_samples"]
    n_train = p["n_train"]
    length = p["length"]
    seed = p["seed"]

    t0 = perf_counter()
    samples = generate_memory_task_dataset(n_samples, length=length, seed=seed)
    train, test = samples[:n_train], samples[n_train:]

    model, _ = build_empirical_model_from_config(config)
    train_seq = np.stack([s.sequence for s in train], axis=0)
    test_seq = np.stack([s.sequence for s in test], axis=0)
    train_y = np.asarray([s.target for s in train], dtype=np.float64)
    test_y = np.asarray([s.target for s in test], dtype=np.float64)

    train_x = exact_summary_matrix(model, train_seq)
    test_x = exact_summary_matrix(model, test_seq)
    mse = ridge_regression_mse(train_x, train_y, test_x, test_y)
    finite = bool(np.isfinite(mse))

    row = {
        "n_train": n_train,
        "n_test": n_samples - n_train,
        "mse": mse,
        "finite": finite,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("train_summary.npy", train_x)
    writer.save_numpy("test_summary.npy", test_x)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "mse": mse,
        "finite": finite,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-11",
            run_experiment_fn=run_experiment,
        )
    )
