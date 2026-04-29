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
    p = config["params"]
    base_seed = p["base_seed"]
    lengths = p["lengths"]
    n_batch = p["n_batch"]

    t0 = perf_counter()
    rows: list[dict[str, Any]] = []

    for length in lengths:
        cfg_copy = dict(config)
        cfg_copy["params"] = dict(p)
        cfg_copy["params"]["seed"] = base_seed + length
        cfg_copy["params"]["length"] = length
        cfg_copy["model"] = dict(config.get("model", {}))
        cfg_copy["model"]["max_history_tokens"] = length

        model, bundle = build_empirical_model_from_config(cfg_copy)
        batch = torch.from_numpy(bundle.test_sequences[:n_batch]).float()

        start = perf_counter()
        _ = model.compute_proxy(batch)
        elapsed_step = perf_counter() - start

        row = {
            "length": length,
            "seconds": elapsed_step,
        }
        rows.append(row)
        writer.log_row("results.jsonl", **row)

    writer.write_csv("results.csv", rows)
    writer.save_numpy("runtimes.npy", np.array([r["seconds"] for r in rows], dtype=np.float64))

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "total_cases": len(rows),
        "all_finite": all(r["seconds"] >= 0.0 for r in rows),
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-12",
            run_experiment_fn=run_experiment,
        )
    )
