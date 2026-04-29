"""Z3 Dataset Adapter Interface.

Defines the contract that every dataset adapter must satisfy, along with
the shared data structures that flow through the multi-dataset pipeline.

Design principles
-----------------
- ``DatasetSpec`` is frozen (immutable, hashable) so it can be used safely
  as a dict key and passed across threads without accidental mutation.
- ``DatasetBundle`` is the adapter's public output — pre-split numpy arrays
  that map directly to ``TrajectoryDataset`` and the training pipeline,
  avoiding an intermediate list-of-samples conversion step.
- ``Z3Adapter`` is the abstract contract.  Each concrete adapter implements
  ``load_splits()`` which returns a ``DatasetBundle``.  The optional
  ``load_samples()`` is provided for consumers that need per-sample access
  (e.g. empirical studies, visualization).

Reference
---------
- Baselines: ``baselines/src/data/adapters/base_adapter.py``
- Z3 plan:   ``docs/implementions/plan.md`` §4.1–4.2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Dataset specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset specification (immutable).

    Captures the dimensions, modality, source, and task metadata required
    to configure the Z3 model and data pipeline for a specific dataset.
    Every field is Z3-relevant — no robotics-specific assumptions.

    Parameters
    ----------
    name : str
        Canonical identifier used in configs, CLI flags, and reports
        (e.g. ``"synthetic"``, ``"telecom"``, ``"spatial"``).
    modality : str
        Data modality — one of ``"temporal"``, ``"geometric"``,
        ``"scientific"``.  Determines which ``Preprocessor`` is applied.
    source : str
        Loading mechanism — ``"huggingface"``, ``"synthetic"``, or
        ``"local"``.
    hf_repo : str or None
        HuggingFace dataset repository ID (required when
        ``source="huggingface"``).
    input_dim : int
        Dimensionality *d* of each observation vector in a sequence.
    sequence_length : int
        Fixed length *T* (or point count *N*) per sample after
        preprocessing (padding / truncation / sub-sampling).
    num_classes : int
        Number of output classes for classification tasks.
    task : str
        Task type — ``"classification"``, ``"anomaly"``, or
        ``"retrieval"``.
    batch_size_override : int or None
        Override the global batch size for this dataset (useful when
        datasets have very different memory footprints).
    max_samples : int or None
        Cap on the number of samples loaded (for smoke tests).
    data_root : str or None
        Root directory for all downloaded dataset files.
        Each dataset is stored under ``<data_root>/<name>/``.
    local_path : str or None
        Path to local data directory or file.
    train_path : str or None
        Explicit path to the training split (overrides default split
        logic).
    val_path : str or None
        Explicit path to the validation split.
    test_path : str or None
        Explicit path to the test split.
    """

    name: str = "synthetic"
    modality: str = "temporal"
    source: str = "synthetic"
    hf_repo: str | None = None
    input_dim: int = 2
    sequence_length: int = 128
    num_classes: int = 4
    task: str = "classification"
    batch_size_override: int | None = None
    max_samples: int | None = None
    data_root: str | None = None
    local_path: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None


# ---------------------------------------------------------------------------
# Dataset bundle — adapter output
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    """Pre-split dataset arrays produced by every adapter.

    This is the primary output of ``Z3Adapter.load_splits()`` and maps
    directly to ``TrajectoryDataset`` without an intermediate conversion
    step.  All arrays are float32 (sequences) or int64 (labels) numpy
    arrays.

    Parameters
    ----------
    train_sequences : np.ndarray
        Shape ``(N_train, T, d)`` — training observations.
    train_labels : np.ndarray
        Shape ``(N_train,)`` — training targets (int64 for
        classification).
    val_sequences : np.ndarray
        Shape ``(N_val, T, d)`` — validation observations.
    val_labels : np.ndarray
        Shape ``(N_val,)`` — validation targets.
    test_sequences : np.ndarray
        Shape ``(N_test, T, d)`` — test observations.
    test_labels : np.ndarray
        Shape ``(N_test,)`` — test targets.
    spec : DatasetSpec
        The specification that produced this bundle.
    metadata : dict
        Optional extra information (topology names, feature names, etc.).
    """

    train_sequences: np.ndarray
    train_labels: np.ndarray
    val_sequences: np.ndarray
    val_labels: np.ndarray
    test_sequences: np.ndarray
    test_labels: np.ndarray
    spec: DatasetSpec = field(default_factory=DatasetSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def train_size(self) -> int:
        return int(self.train_sequences.shape[0])

    @property
    def val_size(self) -> int:
        return int(self.val_sequences.shape[0])

    @property
    def test_size(self) -> int:
        return int(self.test_sequences.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.train_sequences.shape[2])

    @property
    def sequence_length(self) -> int:
        return int(self.train_sequences.shape[1])


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class Z3Adapter(ABC):
    """Abstract base class for all Z3 dataset adapters.

    Each concrete adapter reads a specific data source and produces a
    ``DatasetBundle`` with pre-split numpy arrays.  The adapter also
    exposes a ``DatasetSpec`` so that the training engine can derive the
    correct ``SynapseConfig`` via ``SynapseConfig.for_dataset(spec)``.

    Subclasses must implement:
        - ``load_splits()`` → ``DatasetBundle``
        - ``spec`` property → ``DatasetSpec``
        - ``input_dim`` property → ``int``
        - ``num_classes`` property → ``int``

    Optionally override ``load_samples()`` for per-sample access.
    """

    # ---- required interface ------------------------------------------------

    @abstractmethod
    def load_splits(self) -> DatasetBundle:
        """Load the dataset and return pre-split train/val/test arrays.

        Returns
        -------
        DatasetBundle
            Bundle containing ``train_sequences``, ``train_labels``,
            ``val_sequences``, ``val_labels``, ``test_sequences``,
            ``test_labels``, and the ``spec`` that produced them.
        """
        ...

    @property
    @abstractmethod
    def spec(self) -> DatasetSpec:
        """The ``DatasetSpec`` for this adapter."""
        ...

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimensionality *d* of each observation vector."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""
        ...

    # ---- optional interface ------------------------------------------------

    def load_samples(self) -> list[dict[str, Any]]:
        """Load individual samples as a list of dicts.

        Default implementation unpacks ``load_splits()`` into per-sample
        dicts.  Subclasses may override for more efficient per-sample
        access (e.g. streaming from disk).

        Returns
        -------
        list of dict
            Each dict has keys ``"sequence"`` (np.ndarray, shape
            ``(T, d)``), ``"label"`` (int), ``"split"`` (str).
        """
        bundle = self.load_splits()
        samples: list[dict[str, Any]] = []
        for split_name, seqs, labels in [
            ("train", bundle.train_sequences, bundle.train_labels),
            ("val", bundle.val_sequences, bundle.val_labels),
            ("test", bundle.test_sequences, bundle.test_labels),
        ]:
            for i in range(seqs.shape[0]):
                samples.append({
                    "sequence": seqs[i],
                    "label": int(labels[i]),
                    "split": split_name,
                })
        return samples


__all__ = [
    "DatasetBundle",
    "DatasetSpec",
    "Z3Adapter",
]
