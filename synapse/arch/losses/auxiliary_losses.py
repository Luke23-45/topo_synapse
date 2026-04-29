from __future__ import annotations

import torch


def proxy_regularization(proxy_features: torch.Tensor) -> torch.Tensor:
    return proxy_features.square().mean()


def selector_sparsity(y_star: torch.Tensor) -> torch.Tensor:
    return y_star.mean()
