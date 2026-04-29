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
    n_batch = p["n_batch"]
    n_steps = p["n_steps"]
    learning_rate = p["learning_rate"]

    t0 = perf_counter()
    model, bundle = build_empirical_model_from_config(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch = torch.from_numpy(bundle.train_sequences[:n_batch]).float()
    targets = torch.from_numpy(bundle.train_labels[:n_batch]).long()
    before = model.lift.W_theta.detach().clone()

    loss_history: list[float] = []
    for step in range(n_steps):
        out = model(batch)
        loss = torch.nn.CrossEntropyLoss()(out.logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    delta = float((model.lift.W_theta.detach() - before).norm().item())
    passed = delta > 0.0

    row = {
        "n_steps": n_steps,
        "learning_rate": learning_rate,
        "lift_delta_norm": delta,
        "lift_updates": passed,
    }
    writer.log_row("results.jsonl", **row)
    writer.write_csv("results.csv", [row])

    writer.save_numpy("loss_history.npy", np.array(loss_history, dtype=np.float64))

    elapsed = perf_counter() - t0
    summary = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "formal_reference": config["formal_reference"],
        "claim": config["claim"],
        "duration_seconds": elapsed,
        "lift_delta_norm": delta,
        "lift_updates": passed,
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(
        __import__("synapse.empirical.common.shared", fromlist=["run_standalone"]).run_standalone(
            experiment_id="EZ3-10",
            run_experiment_fn=run_experiment,
        )
    )
