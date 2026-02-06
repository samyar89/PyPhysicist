"""DeepONet operator surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import BaseSurrogate, DatasetLike, SurrogatePrediction, _extract_xy
from ..train import TrainConfig, train_torch_regression
from ..utils import optional_import, torch_available


def _require_torch():
    if not torch_available():
        raise ImportError("PyTorch is required for DeepONet models. Install torch to use this feature.")
    return optional_import("torch", extra="torch")


def _require_torch_nn():
    torch = _require_torch()
    return torch, optional_import("torch.nn", extra="torch"), optional_import("torch.nn.functional", extra="torch")


class DeepONet:  # pragma: no cover - exercised when torch is installed
    def __init__(self, branch_dim: int, trunk_dim: int, width: int, depth: int, activation: str = "gelu"):
        torch, nn, F = _require_torch_nn()
        self.branch = nn.ModuleList([nn.Linear(branch_dim, width)])
        self.trunk = nn.ModuleList([nn.Linear(trunk_dim, width)])
        for _ in range(depth - 1):
            self.branch.append(nn.Linear(width, width))
            self.trunk.append(nn.Linear(width, width))
        self.head = nn.Linear(width, 1)
        self.act = getattr(F, activation)

    def parameters(self):
        params = []
        for layer in self.branch:
            params += list(layer.parameters())
        for layer in self.trunk:
            params += list(layer.parameters())
        params += list(self.head.parameters())
        return params

    def __call__(self, branch_input, trunk_input):
        x = branch_input
        for layer in self.branch:
            x = self.act(layer(x))
        y = trunk_input
        for layer in self.trunk:
            y = self.act(layer(y))
        combined = x * y
        return self.head(combined)


@dataclass
class DeepONetSurrogate(BaseSurrogate):
    config: Optional[dict] = None
    model_: Optional[Any] = None

    def _build_model(self, branch_dim: int, trunk_dim: int) -> Any:
        cfg = self.config or {}
        width = int(cfg.get("width", 64))
        depth = int(cfg.get("depth", 3))
        activation = cfg.get("activation", "gelu")
        return DeepONet(branch_dim=branch_dim, trunk_dim=trunk_dim, width=width, depth=depth, activation=activation)

    def fit(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "DeepONetSurrogate":
        x, y = _extract_xy(dataset, y)
        if x.ndim < 2:
            raise ValueError("DeepONet requires inputs shaped (n_samples, branch_dim + trunk_dim).")
        branch_dim = int(self.config.get("branch_dim", x.shape[1] // 2)) if self.config else x.shape[1] // 2
        trunk_dim = x.shape[1] - branch_dim
        self.model_ = self._build_model(branch_dim, trunk_dim)
        config = TrainConfig.from_kwargs(kwargs)
        train_torch_regression(self.model_, x, y, config=config, deeponet_split=branch_dim)
        return self

    def predict(
        self,
        x: np.ndarray,
        params: Optional[np.ndarray] = None,
        return_uncertainty: bool = False,
        **kwargs: Any,
    ) -> SurrogatePrediction:
        if self.model_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        torch = _require_torch()
        self.model_.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        branch_dim = int(self.config.get("branch_dim", x_tensor.shape[1] // 2)) if self.config else x_tensor.shape[1] // 2
        branch = x_tensor[:, :branch_dim]
        trunk = x_tensor[:, branch_dim:]
        with torch.no_grad():
            pred = self.model_(branch, trunk).cpu().numpy()
        variance = np.var(pred, axis=0, keepdims=True) if return_uncertainty else None
        return SurrogatePrediction(mean=pred, variance=variance, metadata={"model": "DeepONet"})
