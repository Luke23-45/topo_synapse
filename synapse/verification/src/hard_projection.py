from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.synapse_core.selection import hard_select_indices
from synapse.verification.utils._shared import ArtifactWriter


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    K = params_cfg["K"]
    r = params_cfg["r"]
    patterns = [np.array(p, dtype=np.float64) for p in params_cfg["patterns"]]

    t0 = perf_counter()
    rows: list[dict[str, Any]] = []

    for i, y in enumerate(patterns):
        indices = hard_select_indices(y, K=K, r=r)
        ordered = indices == sorted(indices)
        budgeted = len(indices) <= K
        refractory = all(abs(a - b) > r for a, b in zip(indices[:-1], indices[1:])) if len(indices) > 1 else True

        row = {
            "pattern_idx": i,
            "K": K,
            "r": r,
            "indices": indices,
            "ordered": ordered,
            "budgeted": budgeted,
            "refractory_separated": refractory,
        }
        rows.append(row)
        writer.log_row("results.jsonl", **row)

    writer.write_csv("results.csv", rows)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "total_cases": len(rows),
        "all_ordered": all(r_["ordered"] for r_ in rows),
        "all_budgeted": all(r_["budgeted"] for r_ in rows),
        "all_refractory_separated": all(r_["refractory_separated"] for r_ in rows),
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-02",
            run_experiment_fn=run_experiment,
        )
    )
