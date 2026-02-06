"""Lightweight surrogate models for PhysLearn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .base import BaseSurrogate, SurrogatePrediction, _extract_xy


@dataclass
class LinearSurrogate(BaseSurrogate):
    """Linear least-squares surrogate with uncertainty from residual variance."""

    coef_: Optional[np.ndarray] = None
    intercept_: Optional[np.ndarray] = None
    residual_var_: Optional[np.ndarray] = None

    def fit(self, dataset: Any, y: Optional[np.ndarray] = None, **kwargs: Any) -> "LinearSurrogate":
        x, y = _extract_xy(dataset, y)
        if x.ndim == 1:
            x = x[:, None]
        x_aug = np.c_[x, np.ones((x.shape[0], 1))]
        coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        preds = x_aug @ coef
        residuals = y - preds
        self.residual_var_ = np.var(residuals, axis=0, ddof=max(1, x.shape[1]))
        return self

    def predict(self, x: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        mean = x @ self.coef_ + self.intercept_
        std = None
        if self.residual_var_ is not None:
            std = np.sqrt(self.residual_var_)
        variance = None
        if std is not None:
            variance = std**2
        return SurrogatePrediction(mean=mean, variance=variance, metadata={"model": "linear"})


@dataclass
class EnsembleSurrogate(BaseSurrogate):
    """Simple ensemble of linear surrogates for epistemic uncertainty."""

    members: Iterable[LinearSurrogate]
    member_predictions_: Optional[np.ndarray] = None

    def fit(self, dataset: Any, y: Optional[np.ndarray] = None, **kwargs: Any) -> "EnsembleSurrogate":
        x, y = _extract_xy(dataset, y)
        for member in self.members:
            bootstrap_idx = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
            member.fit(x[bootstrap_idx], y[bootstrap_idx])
        return self

    def predict(self, x: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        preds = []
        for member in self.members:
            pred = member.predict(x).mean
            preds.append(pred)
        stacked = np.stack(preds, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        self.member_predictions_ = stacked
        return SurrogatePrediction(mean=mean, variance=std**2, metadata={"model": "ensemble"})


@dataclass
class IdentitySurrogate(BaseSurrogate):
    """Pass-through surrogate for debugging or solver-backed predictions."""

    def fit(self, dataset: Any, y: Optional[np.ndarray] = None, **kwargs: Any) -> "IdentitySurrogate":
        return self

    def predict(self, x: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        x = np.asarray(x)
        return SurrogatePrediction(mean=x, variance=None, metadata={"model": "identity"})


@dataclass
class PlaceholderSurrogate(BaseSurrogate):
    """Placeholder for complex models (FNO, DeepONet, PINN)."""

    name: str
    config: Optional[Dict[str, Any]] = None
    statistics_: Optional[Dict[str, Any]] = None

    def fit(self, dataset: Any, y: Optional[np.ndarray] = None, **kwargs: Any) -> "PlaceholderSurrogate":
        x, y = _extract_xy(dataset, y)
        self.statistics_ = {
            "input_mean": np.mean(x, axis=0),
            "output_mean": np.mean(y, axis=0),
            "samples": x.shape[0],
        }
        return self

    def predict(self, x: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        x = np.asarray(x)
        if self.statistics_ is None:
            raise RuntimeError("Model must be fit before prediction.")
        mean = np.broadcast_to(self.statistics_["output_mean"], (x.shape[0],) + np.asarray(self.statistics_["output_mean"]).shape)
        std = np.std(mean, axis=0)
        return SurrogatePrediction(mean=mean, variance=std**2, metadata={"model": self.name})
