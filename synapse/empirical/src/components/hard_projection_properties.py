from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.empirical.common.shared import ArtifactWriter
from synapse.synapse_core.selection import hard_select_indices


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    K = p["K"]
    r = p["r"]
    pattern = np.array(p["pattern"], dtype=np.float64)

    t0 = perf_counter()
    idx = hard_select_indices(pattern, K=K, r=r)

    ordered = idx == sorted(idx)
    budgeted = len(idx) <= K
    refractory = all(abs(a - b) > r for a, b in zip(idx[:-1], idx[1:])) if len(idx) > 1 else True

    row = {
        "K": K,
        "r": r,
        "indices": idx,
        "ordered": ordered,
        "budgeted": budgeted,
        "refractory_separated": refractory,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("pattern.npy", pattern)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "ordered": ordered,
        "budgeted": budgeted,
        "refractory_separated": refractory,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-08",
            run_experiment_fn=run_experiment,
        )
    )
