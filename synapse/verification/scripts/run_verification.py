from __future__ import annotations

import argparse

import torch

from synapse.common.runtime import load_config
from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.io import save_json
from synapse.verification.checks import verify_absolute_time_invariance, verify_proxy_output_finite
from synapse.verification.stability import topology_stability_trial


def main() -> int:
    parser = argparse.ArgumentParser(description="Run publication-oriented verification checks.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="synapse_outputs/verification.json")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Z3TopologyFirstModel(config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    sequences, _, _ = generate_topology_dataset(1, length=config.max_history_tokens, noise_std=config.noise_std, seed=config.seed + 50)
    sequence = sequences[0]
    report = {}
    report.update(verify_absolute_time_invariance(model, sequence))
    report.update(verify_proxy_output_finite(model, sequence))
    report.update(topology_stability_trial(model, sequence, noise_std=config.noise_std))
    save_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
