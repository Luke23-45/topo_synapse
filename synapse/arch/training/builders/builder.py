"""Model builder for Z3 SYNAPSE training.

Constructs ``Z3TopologyFirstModel`` or ``Z3UnifiedModel`` instances
from configuration objects and resolves lift-layer normalization
statistics from dataset bundles.

The legacy ``build_model_from_cfg()`` builds the original
Z3TopologyFirstModel.  The new ``build_unified_model_from_cfg()``
builds the Z3UnifiedModel that supports backbone switching for
the baseline study.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from synapse.dataset.adapters.base import DatasetBundle
from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import TopologyFirstModel
from synapse.synapse_arch.unified import UnifiedModel

log = logging.getLogger(__name__)


def build_model_from_cfg(cfg, bundle: DatasetBundle | None = None) -> TopologyFirstModel:
    """Construct the active topology-first model from config and bundle spec.

    When a ``DatasetBundle`` is provided, its spec is used to derive
    the correct ``input_dim``, ``output_dim``, and
    ``max_history_tokens`` via ``SynapseConfig.for_dataset()``.
    """
    base_cfg = SynapseConfig(
        input_dim=cfg.model.get("input_dim"),
        output_dim=cfg.model.get("output_dim"),
        hidden_dim=cfg.model.get("hidden_dim"),
        d_model=cfg.model.get("d_model"),
        num_heads=cfg.model.get("num_heads"),
        num_layers=cfg.model.get("num_layers"),
        ffn_ratio=cfg.model.get("ffn_ratio"),
        dropout=cfg.model.get("dropout"),
        K=cfg.model.get("K"),
        r=cfg.model.get("r"),
        lam=cfg.model.get("lam"),
        Q=cfg.model.get("Q"),
        k=cfg.model.get("k"),
        max_history_tokens=cfg.model.get("max_history_tokens"),
        batch_size=cfg.training.get("batch_size"),
        epochs=cfg.training.get("epochs"),
        train_size=cfg.data.get("train_size"),
        val_size=cfg.data.get("val_size"),
        test_size=cfg.data.get("test_size"),
        noise_std=cfg.data.get("noise_std", 0.0),
        seed=cfg.execution.get("seed", 42),
    )
    if bundle is not None:
        base_cfg = base_cfg.for_dataset(bundle.spec)
    return TopologyFirstModel(base_cfg)


def resolve_normalization(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    """Compute structure-aware lift normalization stats from train data."""
    from synapse.arch.data.normalization import compute_normalization_stats

    cached_mu = bundle.metadata.get("normalization_mu")
    cached_sigma = bundle.metadata.get("normalization_sigma")
    if cached_mu is not None and cached_sigma is not None:
        return {
            "mu": np.asarray(cached_mu, dtype=np.float64),
            "sigma": np.asarray(cached_sigma, dtype=np.float64),
        }

    stats = compute_normalization_stats(bundle.train_sequences)
    bundle.metadata["normalization_mu"] = stats["mu"]
    bundle.metadata["normalization_sigma"] = stats["sigma"]
    return stats


def build_unified_model_from_cfg(
    cfg,
    bundle: DatasetBundle | None = None,
) -> UnifiedModel:
    """Construct the active unified model from config and optional bundle spec.

    Supports backbone switching via ``cfg.model.backbone_type``.
    When a ``DatasetBundle`` is provided, its spec is used to derive
    the correct ``input_dim``, ``num_classes``, and sequence length.
    """
    model_cfg = cfg.model
    data_cfg = cfg.data

    input_dim = model_cfg.get("input_dim")
    d_model = model_cfg.get("d_model")
    num_classes = model_cfg.get("output_dim")
    num_tokens = model_cfg.get("max_history_tokens")
    backbone_type = getattr(model_cfg, "backbone_type", "deep_hodge")
    modality = getattr(data_cfg, "modality", "temporal")

    if bundle is not None:
        input_dim = bundle.spec.input_dim
        num_classes = bundle.spec.num_classes
        num_tokens = bundle.spec.sequence_length
        modality = bundle.spec.modality

    model = UnifiedModel(
        backbone_type=backbone_type,
        modality=modality,
        input_dim=input_dim,
        d_model=d_model,
        num_classes=num_classes,
        num_tokens=num_tokens,
        dropout=model_cfg.get("dropout"),
        num_layers=model_cfg.get("num_layers"),
        num_heads=model_cfg.get("num_heads"),
        ffn_ratio=model_cfg.get("ffn_ratio"),
        k_dim=model_cfg.get("k"),
        num_scales=model_cfg.get("num_scales"),
        max_proxy_points=model_cfg.get("max_proxy_points"),
        K=model_cfg.get("K"),
        r=model_cfg.get("r"),
        lam=model_cfg.get("lam"),
        hidden_dim=model_cfg.get("hidden_dim"),
    )

    log.info(
        "Built UnifiedModel: backbone=%s, modality=%s, params=%d",
        backbone_type, modality, model.num_parameters,
    )
    return model


__all__ = ["build_model_from_cfg", "build_unified_model_from_cfg", "resolve_normalization"]
