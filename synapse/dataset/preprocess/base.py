"""Z3 Modality Preprocessors — Base Interface.

Each preprocessor transforms raw samples from a specific modality
(temporal, geometric, scientific) into the uniform ``(T, d)`` sequence
format that the Z3 model consumes.  Preprocessors are composable:
an adapter may chain multiple transforms, and the same preprocessor
can be shared across adapters of the same modality.

Design principles
-----------------
- Preprocessors operate on **numpy arrays** (not torch tensors) so they
  can be applied before dataset construction and device placement.
- Each preprocessor is a pure function wrapped in a class for
  configurability and testability.
- The ``Preprocessor`` ABC defines a single ``__call__`` method that
  transforms a ``DatasetBundle`` into a new ``DatasetBundle`` with
  standardized shape and normalization.

Reference
---------
- Z3 plan: ``docs/implementions/plan.md`` §4.4
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..adapters.base import DatasetBundle


class Preprocessor(ABC):
    """Abstract base for modality-specific preprocessors.

    A preprocessor takes a ``DatasetBundle`` and returns a new
    ``DatasetBundle`` with sequences reshaped, padded, truncated,
    and/or normalized to conform to the Z3 input contract:
    ``sequences.shape == (N, T, d)`` where *T* and *d* match the
    ``DatasetSpec``.
    """

    @abstractmethod
    def __call__(self, bundle: DatasetBundle) -> DatasetBundle:
        """Apply the preprocessing transform to a dataset bundle.

        Parameters
        ----------
        bundle : DatasetBundle
            Raw bundle from an adapter.

        Returns
        -------
        DatasetBundle
            Transformed bundle with standardized shape and normalization.
            The ``spec`` field is updated to reflect any changes in
            ``input_dim`` or ``sequence_length``.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this preprocessor."""
        ...


__all__ = ["Preprocessor"]
