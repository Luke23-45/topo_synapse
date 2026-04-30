from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from synapse.arch.data import data as data_module
from synapse.dataset.adapters.base import DatasetBundle, DatasetSpec


class _DummyAdapter:
    def __init__(self, bundle: DatasetBundle) -> None:
        self._bundle = bundle

    def load_splits(self) -> DatasetBundle:
        return self._bundle


def test_build_dataloaders_respects_training_loader_kwargs(monkeypatch) -> None:
    bundle = DatasetBundle(
        train_sequences=np.zeros((4, 6, 3), dtype=np.float32),
        train_labels=np.zeros(4, dtype=np.int64),
        val_sequences=np.zeros((2, 6, 3), dtype=np.float32),
        val_labels=np.zeros(2, dtype=np.int64),
        test_sequences=np.zeros((2, 6, 3), dtype=np.float32),
        test_labels=np.zeros(2, dtype=np.int64),
        spec=DatasetSpec(name="dummy"),
    )

    captured: list[dict] = []

    def fake_create_adapter(name: str, **kwargs):
        return _DummyAdapter(bundle)

    def fake_dataloader(dataset, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(dataset=dataset, kwargs=kwargs)

    cfg = SimpleNamespace(
        data=SimpleNamespace(dataset="dummy"),
        training=SimpleNamespace(
            batch_size=8,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        ),
    )

    monkeypatch.setattr(data_module, "create_adapter", fake_create_adapter)
    monkeypatch.setattr(data_module, "DataLoader", fake_dataloader)

    train_loader, val_loader, test_loader, _ = data_module.build_dataloaders(cfg)

    assert train_loader.kwargs["batch_size"] == 8
    assert train_loader.kwargs["num_workers"] == 2
    assert train_loader.kwargs["pin_memory"] is True
    assert train_loader.kwargs["persistent_workers"] is True
    assert train_loader.kwargs["prefetch_factor"] == 4
    assert len(captured) == 3
    assert all(loader.kwargs["num_workers"] == 2 for loader in (train_loader, val_loader, test_loader))
