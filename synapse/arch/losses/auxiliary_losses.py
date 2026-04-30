from __future__ import annotations

import torch


def proxy_regularization(proxy_features: torch.Tensor) -> torch.Tensor:
    return proxy_features.square().mean()


def selector_sparsity(y_star: torch.Tensor) -> torch.Tensor:
    # Dense Z4 routing keeps total mass roughly fixed, so mean(y_star) is nearly
    # constant and no longer acts as a useful training signal. Penalize
    # concentration instead to discourage collapse onto a few positions.
    return y_star.square().mean()
