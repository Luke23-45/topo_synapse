from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import torch

from synapse.empirical.common.metrics import match_f1
from synapse.empirical.common.shared import ArtifactWriter, build_empirical_model_from_config


def run_experiment(
    config: dict[str, Any],
    writer: ArtifactWriter,
    verbose: bool = False,
) -> dict[str, Any]:
    p = config["params"]
    length = p["length"]
    changepoints = p["changepoints"]
    match_tolerance = p["match_tolerance"]
    f1_threshold = p["f1_threshold"]

    t0 = perf_counter()
    model, _ = build_empirical_model_from_config(config)

    seq = np.zeros((length, 2), dtype=np.float32)
    for i, cp in enumerate(changepoints):
        seq[cp:] += np.array([1.0, (-1.0) ** i], dtype=np.float32)

    saliency = model(torch.from_numpy(seq).float().unsqueeze(0)).saliency_scores[0].detach().cpu().numpy()
    n_peaks = len(changepoints)
    pred = np.argsort(saliency)[-n_peaks:].tolist()
    score = match_f1(changepoints, pred, tolerance=match_tolerance)
    passed = score > f1_threshold

    row = {
        "changepoints": changepoints,
        "predicted_peaks": pred,
        "match_f1": score,
        "f1_threshold": f1_threshold,
        "alignment_ok": passed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("saliency_scores.npy", saliency)

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "match_f1": score,
        "alignment_ok": passed,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-06",
            run_experiment_fn=run_experiment,
        )
    )
