"""Ablation configuration builder for Z3 SYNAPSE models.

Generates structurally meaningful ablation variants from a base
``SynapseConfig``, allowing systematic evaluation of each component's
contribution.

Ablation variants:
    - baseline: Unmodified base config (control)
    - no_proxy_loss: Zero proxy regularization weight
    - reduced_anchor_budget: Halved anchor budget (K // 2)
    - wider_refractory: Increased refractory separation (+2)
    - no_topology: Zero topology projection (disables Π_top)
    - single_scale: Single spectral scale instead of default
"""

from __future__ import annotations

from copy import deepcopy

from synapse.synapse_arch.config import SynapseConfig


def build_ablation_configs(base: SynapseConfig) -> dict[str, SynapseConfig]:
    """Build ablation config variants from a base configuration.

    Parameters
    ----------
    base : SynapseConfig
        Base (full) configuration.

    Returns
    -------
    dict mapping ablation_name → SynapseConfig
    """
    no_proxy = deepcopy(base)
    no_proxy.proxy_weight = 0.0

    sparse_selector = deepcopy(base)
    sparse_selector.K = max(2, base.K // 2)

    wider_refractory = deepcopy(base)
    wider_refractory.r = base.r + 2

    no_topology = deepcopy(base)
    no_topology.Q = 0

    single_scale = deepcopy(base)
    single_scale.num_scales = 1

    return {
        "baseline": deepcopy(base),
        "no_proxy_loss": no_proxy,
        "reduced_anchor_budget": sparse_selector,
        "wider_refractory": wider_refractory,
        "no_topology": no_topology,
        "single_scale": single_scale,
    }
