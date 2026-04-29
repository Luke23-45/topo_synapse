"""Z3 Baseline Study — Experiment Configuration.

Mirrors the robust pattern from the legacy ``baselines/src/core/config.py``
with frozen dataclasses, proper per-dataset derivation, multi-seed support,
and per-backbone config merging.

Reuses
------
- ``synapse.synapse_arch.config.SynapseConfig`` — model hyperparameters
- ``synapse.synapse.losses.combined_loss.LossConfig`` — loss weighting
"""

from __future__ import annotations

import logging
import warnings
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Directory containing per-backbone overlay YAML files.
BACKBONE_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


# ---------------------------------------------------------------------------
# Backbone conditions (mirrors legacy Condition enum)
# ---------------------------------------------------------------------------

class BackboneCondition(str, Enum):
    """Baseline backbone conditions for controlled comparison."""

    MLP = "mlp"
    TCN = "tcn"
    PTV3 = "ptv3"
    SNN = "snn"
    DEEP_HODGE = "deep_hodge"

    @property
    def label(self) -> str:
        return {
            BackboneCondition.MLP: "MLP (Sanity Check)",
            BackboneCondition.TCN: "TCN (Temporal)",
            BackboneCondition.PTV3: "PTv3 (Geometric)",
            BackboneCondition.SNN: "SNN (Topological)",
            BackboneCondition.DEEP_HODGE: "Deep Hodge (Proposed)",
        }[self]

    @property
    def is_baseline(self) -> bool:
        return self != BackboneCondition.DEEP_HODGE

    @property
    def is_proposed(self) -> bool:
        return self == BackboneCondition.DEEP_HODGE

    @property
    def default_modality(self) -> str:
        """Default modality for this backbone."""
        return {
            BackboneCondition.MLP: "temporal",
            BackboneCondition.TCN: "temporal",
            BackboneCondition.PTV3: "geometric",
            BackboneCondition.SNN: "geometric",
            BackboneCondition.DEEP_HODGE: "topological",
        }[self]

    @property
    def accepted_model_params(self) -> frozenset[str]:
        """ModelParams fields that this backbone actually consumes.

        Fields not in this set are silently ignored when constructing
        the model, preventing ``TypeError`` from unexpected kwargs.
        """
        # Shared params accepted by ALL backbones via Z3UnifiedModel
        _COMMON = frozenset({
            "d_model", "hidden_dim", "dropout",
        })
        return _COMMON | {
            BackboneCondition.MLP: frozenset({"num_tokens"}),
            BackboneCondition.TCN: frozenset({"kernel_size"}),
            BackboneCondition.PTV3: frozenset({
                "num_layers", "num_heads", "ffn_ratio",
                "drop_path", "use_coords",
            }),
            BackboneCondition.SNN: frozenset({
                "num_layers", "ffn_ratio", "k_neighbors",
                "drop_path", "use_coords",
            }),
            BackboneCondition.DEEP_HODGE: frozenset({
                "num_layers", "k", "num_scales", "max_proxy_points",
                "K", "r", "lam",
            }),
        }[self]


# ---------------------------------------------------------------------------
# Per-dataset spec (mirrors legacy DatasetSpec)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Z3DatasetSpec:
    """Per-dataset specification for multi-dataset baseline experiments.

    Each dataset may have different dimensions, modalities, etc.
    These override the global defaults when a specific dataset is active.
    """
    name: str = "synthetic"
    modality: str = "temporal"
    modality_override: Optional[str] = None
    source: str = "synthetic"
    input_dim: int = 2
    sequence_length: int = 128
    num_classes: int = 4
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128
    noise_std: float = 0.03
    batch_size_override: Optional[int] = None
    max_episodes: Optional[int] = None


# ---------------------------------------------------------------------------
# Model params (mirrors legacy TransformerParams + SynapseParams)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelParams:
    """Model architecture parameters — shared across all backbones.

    Backbone-specific params (e.g. k_neighbors for SNN) are set via
    per-backbone YAML config merging in ``for_backbone()``.
    """
    d_model: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_ratio: int = 4
    dropout: float = 0.1
    k: int = 16
    num_scales: int = 3
    max_proxy_points: int = 16
    K: int = 8
    r: int = 1
    lam: float = 0.5
    drop_path: float = 0.1
    k_neighbors: int = 16
    use_coords: bool = True
    kernel_size: int = 3


# ---------------------------------------------------------------------------
# Training params (mirrors legacy TrainingParams)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingParams:
    """Training hyperparameters — identical across all backbones.

    The controlled variable: only the backbone changes.
    Mirrors legacy TrainingParams with all production features.
    """
    max_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 500
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    use_amp: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    compile_model: bool = False
    fused_adamw: bool = True
    save_checkpoints: bool = True


# ---------------------------------------------------------------------------
# Loss params
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossParams:
    """Loss weighting — auto-set per backbone via ``for_backbone()``."""
    proxy_weight: float = 0.0
    sparsity_weight: float = 0.0
    aux_ramp_start: int = 0
    aux_ramp_end: int = 5


# ---------------------------------------------------------------------------
# Statistical testing params (mirrors legacy StatsParams)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatsParams:
    """Statistical rigor parameters for the comparison protocol."""
    num_seeds: int = 3
    significance_level: float = 0.05
    confidence_level: float = 0.95
    num_bootstrap_samples: int = 10000


# ---------------------------------------------------------------------------
# Rollout params (mirrors legacy rollout_steps)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RolloutParams:
    """Rollout (robustness) evaluation parameters.

    Progressive input corruption to measure accuracy degradation,
    analogous to compounding-error rollout in robotics.
    """
    n_steps: int = 10
    noise_scale: float = 0.1
    max_samples: int = 50


# ---------------------------------------------------------------------------
# Experiment config (mirrors legacy ExperimentConfig)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Z3ExperimentConfig:
    """Complete experiment configuration for the Z3 baseline study.

    Mirrors the structure of the legacy ``ExperimentConfig`` with
    frozen dataclasses and proper ``for_dataset()`` / ``for_backbone()``
    derivation methods.

    Reuses SynapseConfig fields for model hyperparameters.
    """
    backbone: BackboneCondition = BackboneCondition.MLP
    seed: int = 42
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    loss: LossParams = field(default_factory=LossParams)
    stats: StatsParams = field(default_factory=StatsParams)
    rollout: RolloutParams = field(default_factory=RolloutParams)
    datasets: Tuple[Z3DatasetSpec, ...] = field(default_factory=lambda: (
        Z3DatasetSpec(),
    ))
    output_dir: str = "outputs/baselines"
    experiment_name: str = "z3_baseline_study"

    # ------------------------------------------------------------------
    # Derived configs
    # ------------------------------------------------------------------

    def for_backbone(self, backbone: BackboneCondition) -> Z3ExperimentConfig:
        """Create a copy with a different backbone, merging its overlay YAML.

        Resolution: base config → backbone overlay → (caller chains
        ``for_dataset()`` and ``for_seed()`` afterwards).

        The overlay YAML (e.g. ``configs/deep_hodge.yaml``) contains only
        the fields that differ from the base experiment config.  This
        replaces the previous hardcoded if/else loss-weight logic with
        a data-driven, extensible mechanism.
        """
        overlay = load_backbone_overlay(backbone.value)
        if overlay is None:
            # No overlay file exists — just swap the backbone enum.
            return Z3ExperimentConfig(
                backbone=backbone,
                seed=self.seed,
                model=self.model,
                training=self.training,
                loss=self.loss,
                stats=self.stats,
                rollout=self.rollout,
                datasets=self.datasets,
                output_dir=self.output_dir,
                experiment_name=self.experiment_name,
            )

        # Deep-merge overlay into the current config's dict representation.
        base_dict = self.to_dict()
        merged = _deep_merge(base_dict, overlay)
        # Ensure the backbone enum is set correctly (overlay may carry
        # the string name, but we always use the caller's enum).
        merged["backbone"] = backbone.value
        return _config_from_dict(merged)

    def for_dataset(self, dataset_spec: Z3DatasetSpec) -> Z3ExperimentConfig:
        """Create a dataset-specific config by overriding dimensions.

        This is critical for multi-dataset experiments where each dataset
        has different input_dim, num_classes, sequence_length, etc.
        Also propagates batch_size_override into training params.
        """
        if dataset_spec.batch_size_override is not None:
            new_training = TrainingParams(
                max_epochs=self.training.max_epochs,
                batch_size=dataset_spec.batch_size_override,
                learning_rate=self.training.learning_rate,
                weight_decay=self.training.weight_decay,
                beta1=self.training.beta1,
                beta2=self.training.beta2,
                warmup_steps=self.training.warmup_steps,
                early_stopping_patience=self.training.early_stopping_patience,
                gradient_clip_norm=self.training.gradient_clip_norm,
                use_amp=self.training.use_amp,
                num_workers=self.training.num_workers,
                pin_memory=self.training.pin_memory,
                persistent_workers=self.training.persistent_workers,
                prefetch_factor=self.training.prefetch_factor,
                compile_model=self.training.compile_model,
                fused_adamw=self.training.fused_adamw,
                save_checkpoints=self.training.save_checkpoints,
            )
        else:
            new_training = self.training

        return Z3ExperimentConfig(
            backbone=self.backbone,
            seed=self.seed,
            model=self.model,
            training=new_training,
            loss=self.loss,
            stats=self.stats,
            rollout=self.rollout,
            datasets=(dataset_spec,),
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
        )

    def for_seed(self, seed: int) -> Z3ExperimentConfig:
        """Create a copy with a different seed for multi-seed runs."""
        return Z3ExperimentConfig(
            backbone=self.backbone,
            seed=seed,
            model=self.model,
            training=self.training,
            loss=self.loss,
            stats=self.stats,
            rollout=self.rollout,
            datasets=self.datasets,
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
        )

    @property
    def modality(self) -> str:
        """Active modality — explicit override > backbone default > dataset default.

        Resolution order:
            1. ``modality_override`` on the active dataset (explicit per-dataset
               pinning, e.g. a geometric dataset forced to geometric even for
               temporal backbones).
            2. ``backbone.default_modality`` (each backbone knows what it needs:
               PTv3→geometric, SNN→geometric, Deep Hodge→topological,
               MLP/TCN→temporal).
            3. ``dataset.modality`` as a last-resort fallback when no backbone
               is yet selected.
        """
        if self.datasets:
            ds = self.datasets[0]
            if ds.modality_override is not None:
                return ds.modality_override
        return self.backbone.default_modality

    def filtered_model_kwargs(self) -> dict:
        """Return model kwargs filtered to only those the current backbone accepts.

        Prevents passing incompatible parameters (e.g. ``k_neighbors`` to MLP)
        to ``Z3UnifiedModel``.  Dataset-derived dimensions (input_dim,
        num_classes, num_tokens) are always included.
        """
        accepted = self.backbone.accepted_model_params
        model_dict = asdict(self.model)

        # Dataset-derived dimensions are always required
        ds = self.datasets[0] if self.datasets else None
        always = {
            "input_dim": ds.input_dim if ds else 2,
            "num_classes": ds.num_classes if ds else 4,
            "num_tokens": ds.sequence_length if ds else 128,
        }

        filtered = {
            k: v for k, v in model_dict.items()
            if k in accepted
        }
        filtered.update(always)

        # Bridge: ModelParams.k → Z3UnifiedModel parameter k_dim
        # (The unified model uses "k_dim" for the geometric lift dimension
        # while ModelParams calls it "k".)
        if "k" in filtered:
            filtered["k_dim"] = filtered.pop("k")

        return filtered

    def to_dict(self) -> dict:
        d = asdict(self)
        d["backbone"] = self.backbone.value
        d["datasets"] = [asdict(ds) for ds in self.datasets]
        return d

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Config loading (mirrors legacy load_config)
# ---------------------------------------------------------------------------

def load_experiment_config(path: str | Path) -> Z3ExperimentConfig:
    """Load experiment configuration from a YAML file.

    The YAML file may specify any subset of fields; unspecified fields
    retain their defaults.  Mirrors the legacy ``load_config()`` pattern.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _config_from_dict(raw)


def _config_from_dict(raw: dict) -> Z3ExperimentConfig:
    """Construct Z3ExperimentConfig from a nested dictionary."""

    def _pop_nested(d: dict, key: str, default_factory):
        sub = d.pop(key, {})
        if sub is None:
            sub = {}
        # Convert string numbers to floats/ints
        for k, v in sub.items():
            if isinstance(v, str):
                try:
                    sub[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                except ValueError:
                    pass
        return default_factory(**sub)

    backbone_raw = raw.pop("backbone", "mlp")
    try:
        backbone = BackboneCondition(backbone_raw)
    except ValueError:
        valid = sorted(b.value for b in BackboneCondition)
        raise ValueError(
            f"Invalid backbone '{backbone_raw}' in config. "
            f"Valid options: {valid}"
        ) from None
    seed = int(raw.pop("seed", 42))
    output_dir = raw.pop("output_dir", "outputs/baselines")
    experiment_name = raw.pop("experiment_name", "z3_baseline_study")

    model = _pop_nested(raw, "model", ModelParams)
    training = _pop_nested(raw, "training", TrainingParams)
    loss = _pop_nested(raw, "loss", LossParams)
    stats = _pop_nested(raw, "stats", StatsParams)
    rollout = _pop_nested(raw, "rollout", RolloutParams)

    # Parse datasets list
    datasets_raw = raw.pop("datasets", None)
    if datasets_raw is not None:
        datasets = tuple(Z3DatasetSpec(**ds) for ds in datasets_raw)
    else:
        datasets = (Z3DatasetSpec(),)

    if raw:
        warnings.warn(f"Unrecognized config fields ignored: {list(raw.keys())}")

    return Z3ExperimentConfig(
        backbone=backbone,
        seed=seed,
        model=model,
        training=training,
        loss=loss,
        stats=stats,
        rollout=rollout,
        datasets=datasets,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )


# ---------------------------------------------------------------------------
# Backbone overlay loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into a copy of *base*.

    - Dict values are merged recursively (overlay wins on conflicts).
    - Non-dict values from overlay replace base values.
    - base is never mutated; a new dict is returned.
    """
    result = dict(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_backbone_overlay(backbone_name: str) -> dict | None:
    """Load a backbone-specific overlay YAML from ``BACKBONE_CONFIG_DIR``.

    Returns the parsed dict, or ``None`` if no overlay file exists
    (which is valid — the base config is used as-is).
    """
    overlay_path = BACKBONE_CONFIG_DIR / f"{backbone_name}.yaml"
    if not overlay_path.exists():
        log.debug("No backbone overlay at %s — using base config only", overlay_path)
        return None
    with open(overlay_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    log.debug("Loaded backbone overlay: %s (%d keys)", overlay_path.name, len(raw))
    return raw


__all__ = [
    "BackboneCondition",
    "Z3DatasetSpec",
    "ModelParams",
    "TrainingParams",
    "LossParams",
    "StatsParams",
    "RolloutParams",
    "Z3ExperimentConfig",
    "load_experiment_config",
    "load_backbone_overlay",
    "BACKBONE_CONFIG_DIR",
]
