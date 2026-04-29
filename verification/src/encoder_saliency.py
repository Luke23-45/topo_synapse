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
    prefix = params_cfg["prefix"]
    perturbation = params_cfg["perturbation"]
    atol = params_cfg["atol"]
    seed = params_cfg["seed"]
    input_dim = params_cfg["input_dim"]
    max_history_tokens = params_cfg["max_history_tokens"]

    t0 = perf_counter()

    model_cfg = make_config_from_yaml(config["experiment_id"])
    model = build_model(model_cfg)
    seq = random_walk(input_dim, max_history_tokens, seed=seed)

    alt = seq.copy()
    alt[prefix:] += perturbation

    base_scores = model(torch.from_numpy(seq).float().unsqueeze(0)).saliency_scores[0, :prefix]
    alt_scores = model(torch.from_numpy(alt).float().unsqueeze(0)).saliency_scores[0, :prefix]

    max_diff = float((base_scores - alt_scores).abs().max())
    close = bool(torch.allclose(base_scores, alt_scores, atol=atol))

    row = {
        "prefix": prefix,
        "perturbation": perturbation,
        "atol": atol,
        "seed": seed,
        "max_diff": max_diff,
        "prefix_causality_holds": close,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("base_saliency.npy", base_scores.detach().numpy())
    writer.save_numpy("alt_saliency.npy", alt_scores.detach().numpy())

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "prefix_causality_holds": close,
        "max_saliency_diff": max_diff,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.verification.utils._shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="VZ3-03",
            run_experiment_fn=run_experiment,
        )
    )
