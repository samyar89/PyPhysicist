"""Hybrid surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .base import BaseSurrogate, DatasetLike, SurrogatePrediction, _extract_xy
from ..train import TrainConfig, train_torch_regression
from ..utils import optional_import, torch_available


def _require_torch():
    if not torch_available():
        raise ImportError("PyTorch is required for hybrid surrogates. Install torch to use this feature.")
    return optional_import("torch", extra="torch")


@dataclass
class NeuralCorrectorSurrogate(BaseSurrogate):
    """Neural corrector that learns residuals over a coarse solver."""

    coarse_solver: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
    config: Optional[dict] = None
    model_: Optional[Any] = None

    def _build_model(self, input_dim: int, output_dim: int) -> Any:
        torch = _require_torch()
        nn = optional_import("torch.nn", extra="torch")
        cfg = self.config or {}
        width = int(cfg.get("width", 64))
        depth = int(cfg.get("depth", 3))
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, output_dim))
        return nn.Sequential(*layers)

    def fit(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "NeuralCorrectorSurrogate":
        x, y = _extract_xy(dataset, y)
        coarse = self.coarse_solver(x, None)
        residual = y - coarse
        self.model_ = self._build_model(x.shape[1], residual.shape[1] if residual.ndim > 1 else 1)
        config = TrainConfig.from_kwargs(kwargs)
        callbacks = kwargs.get("callbacks")
        train_torch_regression(self.model_, x, residual, config=config, callbacks=callbacks)
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
            residual = self.model_(x_tensor).cpu().numpy()
        coarse = self.coarse_solver(x, params)
        pred = coarse + residual
        variance = np.var(residual, axis=0, keepdims=True) if return_uncertainty else None
        return SurrogatePrediction(mean=pred, variance=variance, metadata={"model": "HybridCorrector"})
