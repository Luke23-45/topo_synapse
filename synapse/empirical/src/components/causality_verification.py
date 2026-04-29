from __future__ import annotations

from time import perf_counter
from typing import Any

import torch

from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    prefix = p["prefix"]
    perturbation = p["perturbation"]
    atol = p["atol"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    seq = bundle.test_sequences[0].copy()
    alt = seq.copy()
    alt[prefix:] += perturbation

    base = model(torch.from_numpy(seq).float().unsqueeze(0)).saliency_scores[0, :prefix]
    pert = model(torch.from_numpy(alt).float().unsqueeze(0)).saliency_scores[0, :prefix]

    max_diff = float((base - pert).abs().max())
    close = bool(torch.allclose(base, pert, atol=atol))

    row = {
        "prefix": prefix,
        "perturbation": perturbation,
        "atol": atol,
        "max_diff": max_diff,
        "prefix_causality_holds": close,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("base_saliency.npy", base.detach().cpu().numpy())
    writer.save_numpy("pert_saliency.npy", pert.detach().cpu().numpy())

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "prefix_causality_holds": close,
        "max_diff": max_diff,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-05",
            run_experiment_fn=run_experiment,
        )
    )
