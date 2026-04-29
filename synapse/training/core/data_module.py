"""Synapse LightningDataModule — data pipeline for Z3 training.

Wraps the existing ``build_dataloaders()`` function in a
LightningDataModule so that the Lightning Trainer can manage
data loading, including distributed sampler setup automatically.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from synapse.dataset.adapters.base import DatasetBundle
from synapse.synapse.data.data import build_dataloaders as _build_dataloaders

log = logging.getLogger(__name__)


class SynapseDataModule(pl.LightningDataModule):
    """LightningDataModule for Z3 SYNAPSE datasets.

    Parameters
    ----------
    cfg : Any
        Configuration object with ``model``, ``data``, ``training``,
        ``execution`` attributes (OmegaConf or dataclass).
    dataset_name : str or None
        Dataset name from the adapter registry.  Falls back to
        ``cfg.data.dataset`` or ``"synthetic"``.
    batch_size : int or None
        Override batch size.  If ``None``, reads from config.
    num_workers : int
        Number of DataLoader workers.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        dataset_name: str | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self._cfg = cfg
        self._dataset_name = dataset_name
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        # Populated in setup()
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.bundle: DatasetBundle | None = None

    def prepare_data(self) -> None:
        """Download/preprocess data if needed (called on single GPU).

        The adapter registry handles data downloading internally,
        so we just call build_dataloaders once to trigger it.
        """
        # No-op: adapters handle their own downloading in load_splits()
        pass

    def setup(self, stage: str | None = None) -> None:
        """Build dataloaders and bundle.

        Called on every process in distributed training.
        """
        if self.train_loader is not None:
            return  # Already set up

        self.train_loader, self.val_loader, self.test_loader, self.bundle = (
            _build_dataloaders(self._cfg, dataset_name=self._dataset_name)
        )

        # Override batch size if specified
        if self._batch_size is not None:
            for loader in (self.train_loader, self.val_loader, self.test_loader):
                if loader is not None:
                    loader.batch_sampler.batch_size = self._batch_size

        log.info(
            "DataModule setup: train=%d, val=%d, test=%d, input_dim=%d",
            self.bundle.train_size,
            self.bundle.val_size,
            self.bundle.test_size,
            self.bundle.input_dim,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_loader
