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
from synapse.synapse_arch.model import Z3TopologyFirstModel
from synapse.synapse_arch.unified import Z3UnifiedModel

log = logging.getLogger(__name__)


def build_model_from_cfg(cfg, bundle: DatasetBundle | None = None) -> Z3TopologyFirstModel:
    """Construct a Z3TopologyFirstModel from config and optional bundle spec.

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
    return Z3TopologyFirstModel(base_cfg)


def resolve_normalization(bundle: DatasetBundle) -> dict[str, np.ndarray]:
    """Compute lift-layer normalization stats, respecting preprocessor stats.

    If the adapter's preprocessor already z-score normalized the data
    (indicated by ``normalization_mean`` / ``normalization_std`` in the
    bundle metadata), those stats are reused for the data dimensions so
    that the lift layer does not double-normalize.  The augmentation
    channels (time, delta, saliency) are always computed fresh.

    For bundles without preprocessor stats (e.g. synthetic), stats are
    computed from the raw training sequences as before.
    """
    from synapse.arch.data.normalization import compute_normalization_stats

    pre_mu = bundle.metadata.get("normalization_mean")
    pre_sigma = bundle.metadata.get("normalization_std")

    if pre_mu is not None and pre_sigma is not None:
        log.info("Using preprocessor normalization stats (skipping recomputation)")
        d = bundle.train_sequences.shape[2]
        T = bundle.train_sequences.shape[1]
        N = bundle.train_sequences.shape[0]

        time_mu = 0.0
        time_sigma = 1.0

        delta_vals = np.ones((N, T), dtype=np.float64)
        delta_vals[:, 0] = 0.0
        delta_mu = float(delta_vals.mean())
        delta_sigma = float(delta_vals.std())
        if delta_sigma <= 0:
            delta_sigma = 1.0

        saliency_mu = 0.0
        saliency_sigma = 1.0

        mu = np.concatenate([
            np.array([time_mu], dtype=np.float64),
            pre_mu.astype(np.float64),
            np.array([delta_mu], dtype=np.float64),
            np.array([saliency_mu], dtype=np.float64),
        ])
        sigma = np.concatenate([
            np.array([time_sigma], dtype=np.float64),
            pre_sigma.astype(np.float64),
            np.array([delta_sigma], dtype=np.float64),
            np.array([saliency_sigma], dtype=np.float64),
        ])
        sigma[sigma <= 0] = 1.0
        return {"mu": mu, "sigma": sigma}

    return compute_normalization_stats(bundle.train_sequences)


def build_unified_model_from_cfg(
    cfg,
    bundle: DatasetBundle | None = None,
) -> Z3UnifiedModel:
    """Construct a Z3UnifiedModel from config and optional bundle spec.

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

    model = Z3UnifiedModel(
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
        "Built Z3UnifiedModel: backbone=%s, modality=%s, params=%d",
        backbone_type, modality, model.num_parameters,
    )
    return model


__all__ = ["build_model_from_cfg", "build_unified_model_from_cfg", "resolve_normalization"]
