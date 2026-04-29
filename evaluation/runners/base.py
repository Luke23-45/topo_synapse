"""Base evaluator interface and shared result types.

Defines the contract that every modality-specific evaluator must satisfy,
along with the ``EvalResult`` dataclass that standardizes evaluation
output across all data types.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from synapse.synapse_arch.config import SynapseConfig
from synapse.synapse_arch.model import Z3TopologyFirstModel


@dataclass
class EvalResult:
    """Standardized evaluation result across all modalities.

    Attributes
    ----------
    dataset_name : str
        Canonical dataset identifier.
    modality : str
        Data modality (``"temporal"``, ``"geometric"``, ``"scientific"``).
    task : str
        Task type (``"classification"``, ``"anomaly"``, ``"retrieval"``).
    primary_metrics : dict[str, float]
        Core task metrics (e.g. accuracy, loss).
    per_sample_metrics : dict[str, np.ndarray]
        Per-sample arrays keyed by metric name.
    topology_metrics : dict[str, float]
        Proxy-exact alignment and topology-specific metrics.
    robustness_metrics : dict[str, Any]
        Robustness sweep results (noise, length, etc.).
    metadata : dict[str, Any]
        Additional context (config snapshot, run info).
    """

    dataset_name: str
    modality: str
    task: str
    primary_metrics: dict[str, float] = field(default_factory=dict)
    per_sample_metrics: dict[str, np.ndarray] = field(default_factory=dict)
    topology_metrics: dict[str, float] = field(default_factory=dict)
    robustness_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        result: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "task": self.task,
            "primary_metrics": self.primary_metrics,
            "topology_metrics": self.topology_metrics,
            "robustness_metrics": self.robustness_metrics,
            "metadata": self.metadata,
        }
        # Convert numpy arrays to lists for JSON serialization
        for key, arr in self.per_sample_metrics.items():
            result[f"per_sample_{key}"] = arr.tolist() if isinstance(arr, np.ndarray) else arr
        return result

    def save_json(self, path: str | Path) -> Path:
        """Save the result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class BaseEvaluator(ABC):
    """Abstract base class for all Z3 evaluators.

    Subclasses must implement ``evaluate()`` which returns an ``EvalResult``.

    Parameters
    ----------
    config : SynapseConfig
        Model configuration.
    test_loader : DataLoader
        Held-out test set.
    output_dir : Path
        Directory for evaluation artifacts (plots, JSON, etc.).
    """

    def __init__(
        self,
        config: SynapseConfig,
        test_loader: DataLoader,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def evaluate(self, model: Z3TopologyFirstModel) -> EvalResult:
        """Run the full evaluation protocol and return results.

        Parameters
        ----------
        model : Z3TopologyFirstModel
            Trained model to evaluate.

        Returns
        -------
        EvalResult
        """
        ...

    def _compute_proxy_exact_alignment(
        self,
        model: Z3TopologyFirstModel,
        sequences: torch.Tensor,
    ) -> dict[str, float]:
        """Compute proxy-exact topology alignment for a subset of samples.

        Parameters
        ----------
        model : Z3TopologyFirstModel
        sequences : torch.Tensor
            Subset of test sequences on device.

        Returns
        -------
        dict with alignment metric.
        """
        from synapse.evaluation.metrics.topology import proxy_exact_alignment

        with torch.no_grad():
            out = model(sequences)

        audits = model.exact_audit(sequences.cpu())
        exact_summaries = np.stack(
            [audit.topology_summary for audit in audits], axis=0
        )
        return proxy_exact_alignment(
            out.proxy_features.detach().cpu().numpy(),
            exact_summaries,
        )

    def _extract_test_samples(
        self,
        max_samples: int = 64,
    ) -> torch.Tensor:
        """Extract a flat tensor of test sequences for topology analysis.

        Parameters
        ----------
        max_samples : int
            Maximum number of samples to extract.

        Returns
        -------
        torch.Tensor, shape (min(N, max_samples), T, d)
        """
        all_seqs = []
        for batch in self.test_loader:
            seqs = batch["sequences"] if "sequences" in batch else batch["sequence"]
            all_seqs.append(seqs)
            if len(all_seqs) * seqs.shape[0] >= max_samples:
                break
        all_seqs = torch.cat(all_seqs, dim=0)[:max_samples]
        return all_seqs.to(self.device)
