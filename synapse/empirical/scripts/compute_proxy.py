from __future__ import annotations

import argparse

import torch

from synapse.common.runtime import load_config
from synapse.empirical.datasets.synthetic_topology import generate_topology_dataset
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.utils.io import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute training-time spectral proxy features.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="synapse_outputs/proxy.json")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = Z3TopologyFirstModel(config)
    model.load_state_dict(payload["model_state"])
    model.eval()

    sequences, _, _ = generate_topology_dataset(2, length=config.max_history_tokens, noise_std=config.noise_std, seed=config.seed + 30)
    proxy = model.compute_proxy(torch.from_numpy(sequences).float())
    save_json(args.output, {"proxy_features": proxy.proxy_features.detach().cpu().tolist()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
