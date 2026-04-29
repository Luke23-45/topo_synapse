from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from synapse.synapse_core.selection import solve_relaxed_selector
from synapse.verification.utils._shared import ArtifactWriter, iter_parameter_grid


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    grid_axes = params_cfg["grid"]
    lam = params_cfg["lam"]
    solver = params_cfg["solver"]
    tolerance = params_cfg["tolerance"]

    t0 = perf_counter()
    rows: list[dict[str, Any]] = []

    for params in iter_parameter_grid(**grid_axes):
        scores = np.linspace(0.0, 1.0, params["T"], dtype=np.float64)
        y = solve_relaxed_selector(scores, K=params["K"], r=params["r"], lam=lam, solver=solver)

        refractory_ok = True
        for offset in range(1, params["r"] + 1):
            if offset < y.size and np.any(y[:-offset] + y[offset:] > 1.0 + tolerance):
                refractory_ok = False

        box_ok = bool(np.all((0.0 <= y) & (y <= 1.0)))
        budget_ok = bool(y.sum() <= params["K"] + tolerance)
        first_step_ok = bool(y[0] == 0.0)

        row = {
            **params,
            "y_sum": float(y.sum()),
            "box_ok": box_ok,
            "budget_ok": budget_ok,
            "first_step_ok": first_step_ok,
            "refractory_ok": refractory_ok,
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
        "all_box_ok": all(r["box_ok"] for r in rows),
        "all_budget_ok": all(r["budget_ok"] for r in rows),
        "all_first_step_ok": all(r["first_step_ok"] for r in rows),
        "all_refractory_ok": all(r["refractory_ok"] for r in rows),
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-01",
            run_experiment_fn=run_experiment,
        )
    )
