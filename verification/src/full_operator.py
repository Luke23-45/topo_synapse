from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from synapse.verification.utils._shared import ArtifactWriter, build_model, iter_parameter_grid, make_config_from_yaml, random_walk


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    params_cfg = config["params"]
    grid_axes = params_cfg["grid"]
    input_dim = params_cfg["input_dim"]
    seed_offset = params_cfg["seed_offset"]

    t0 = perf_counter()
    rows: list[dict[str, Any]] = []

    for params in iter_parameter_grid(**grid_axes):
        model_cfg = make_config_from_yaml(config["experiment_id"], K_override=params["K"], r_override=params["r"], max_history_tokens_override=params["T"])
        model = build_model(model_cfg)
        sequence = random_walk(input_dim, params["T"], seed=params["T"] + params["K"] + seed_offset)

        tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        audit = model.exact_audit(tensor)[0]
        proxy = model.compute_proxy(tensor)

        anchor_bounded = len(audit.anchor_indices) <= model_cfg.K
        proxy_batch_ok = proxy.proxy_features.shape[0] == 1

        row = {
            **params,
            "anchor_count": len(audit.anchor_indices),
            "K_limit": model_cfg.K,
            "anchor_bounded": anchor_bounded,
            "proxy_batch_ok": proxy_batch_ok,
            "proxy_shape": list(proxy.proxy_features.shape),
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
        "all_anchor_bounded": all(r["anchor_bounded"] for r in rows),
        "all_proxy_batch_ok": all(r["proxy_batch_ok"] for r in rows),
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-04",
            run_experiment_fn=run_experiment,
        )
    )
