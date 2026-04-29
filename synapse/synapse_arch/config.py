from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapse.dataset.adapters.base import DatasetSpec


@dataclass
class SynapseConfig:
    """Z3 Topology-First configuration.

    Structural parameters from §3 of 01_main_definition.md:
        K: maximum number of retained deployment anchors
        r: refractory separation between retained anchors
        Q: maximum persistent-homology degree retained by the exact topology audit
        k: latent geometric dimension of the learned lift
        num_scales (S): number of spectral scales
        num_eigs (J): number of ordered eigenvalues retained per operator and per scale
        tau: spectral ridge used in the proxy branch (τ > 0)
    """

    # Data shape
    input_dim: int = 2
    output_dim: int = 4

    # Network architecture
    hidden_dim: int = 64
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_ratio: int = 4
    dropout: float = 0.1

    # Anchor selection (§5–6 of 01_main_definition.md)
    K: int = 8
    r: int = 1
    lam: float = 0.5

    # Topology (§9, §12 of 01_main_definition.md)
    topology_mode: str = "baseline_proxy"  # Options: "baseline_proxy", "deep_hodge", "transformer"
    Q: int = 1
    k: int = 16
    num_scales: int = 3       # S: number of Gaussian kernel scales
    num_eigs: int = 4         # J: eigenvalues per operator per scale
    max_proxy_points: int = 16  # MAX_K: simplicial complex size
    tau: float = 1e-4         # τ: ridge regularization for proxy Laplacians

    # Sequence
    max_history_tokens: int = 128

    # Task
    task: str = "classification"
    loss: str = "cross_entropy"
    proxy_weight: float = 1.0

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128
    noise_std: float = 0.03
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe attribute access with default value."""
        return getattr(self, key, default)

    def for_dataset(self, spec: DatasetSpec) -> SynapseConfig:
        """Create a dataset-specific config by overriding data-shape fields.

        Preserves all Z3 structural parameters (K, r, Q, k, tau, etc.)
        and only replaces the fields that vary per-dataset: input
        dimension, output classes, sequence length, and task type.

        Parameters
        ----------
        spec : DatasetSpec
            Per-dataset specification from the adapter registry.

        Returns
        -------
        SynapseConfig
            New config instance with dataset-appropriate dimensions.
        """
        overrides: dict = {}
        if spec.input_dim != self.input_dim:
            overrides["input_dim"] = spec.input_dim
        if spec.num_classes != self.output_dim:
            overrides["output_dim"] = spec.num_classes
        if spec.sequence_length != self.max_history_tokens:
            overrides["max_history_tokens"] = spec.sequence_length
        if spec.task != self.task:
            overrides["task"] = spec.task
        if spec.batch_size_override is not None:
            overrides["batch_size"] = spec.batch_size_override

        if not overrides:
            return self

        current = asdict(self)
        current.update(overrides)
        return SynapseConfig(**current)
