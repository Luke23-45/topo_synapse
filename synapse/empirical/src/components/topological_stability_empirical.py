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
    noise_perturbation_std = p["noise_perturbation_std"]
    noise_seed = p["noise_seed"]
    n_samples = p["n_samples"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    noisy = bundle.test_sequences + np.random.default_rng(noise_seed).normal(
        scale=noise_perturbation_std, size=bundle.test_sequences.shape
    )
    clean_summary = exact_summary_matrix(model, bundle.test_sequences[:n_samples])
    noisy_summary = exact_summary_matrix(model, noisy[:n_samples])
    delta = float(np.mean(np.abs(clean_summary - noisy_summary)))
    finite = bool(np.isfinite(delta))

    row = {
        "noise_perturbation_std": noise_perturbation_std,
        "mean_abs_delta": delta,
        "finite": finite,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("clean_summary.npy", clean_summary)
    writer.save_numpy("noisy_summary.npy", noisy_summary)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "mean_abs_delta": delta,
        "finite": finite,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-02",
            run_experiment_fn=run_experiment,
        )
    )
