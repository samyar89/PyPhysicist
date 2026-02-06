"""Physics-Informed Neural Network surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .base import BaseSurrogate, DatasetLike, SurrogatePrediction, _extract_xy
from ..train import TrainConfig, train_pinn
from ..utils import optional_import, torch_available


def _require_torch():
    if not torch_available():
        raise ImportError("PyTorch is required for PINN models. Install torch to use this feature.")
    return optional_import("torch", extra="torch")


@dataclass
class PINNSurrogate(BaseSurrogate):
    """PINN surrogate with residual, data, and boundary losses."""

    config: Optional[dict] = None
    model_: Optional[Any] = None
    residual_fn: Optional[Callable[..., Any]] = None

    def _build_model(self, input_dim: int, output_dim: int) -> Any:
        torch = _require_torch()
        nn = optional_import("torch.nn", extra="torch")
        cfg = self.config or {}
        width = int(cfg.get("width", 64))
        depth = int(cfg.get("depth", 4))
        layers = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, output_dim))
        return nn.Sequential(*layers)

    def fit(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "PINNSurrogate":
        x, y = _extract_xy(dataset, y)
        self.model_ = self._build_model(x.shape[1], y.shape[1] if y.ndim > 1 else 1)
        config = TrainConfig.from_kwargs(kwargs)
        train_pinn(self.model_, x, y, residual_fn=self.residual_fn, config=config, **kwargs)
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
        with torch.no_grad():
            pred = self.model_(x_tensor).cpu().numpy()
        variance = np.var(pred, axis=0, keepdims=True) if return_uncertainty else None
        return SurrogatePrediction(mean=pred, variance=variance, metadata={"model": "PINN"})
