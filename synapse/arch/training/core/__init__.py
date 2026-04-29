"""Core training pipeline for Z3 SYNAPSE models.

Provides the Lightning-based training loop, LightningModule wrapper,
and LightningDataModule for data loading.
"""

from .data_module import SynapseDataModule
from .loop import train
from .module import SynapseLightningModule

__all__ = ["SynapseDataModule", "SynapseLightningModule", "train"]
