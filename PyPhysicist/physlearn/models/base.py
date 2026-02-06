"""Base interfaces for surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np


@dataclass
class SurrogatePrediction:
    mean: np.ndarray
    variance: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def std(self) -> Optional[np.ndarray]:
        if self.variance is None:
            return None
        return np.sqrt(self.variance)


DatasetLike = Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]


def _extract_xy(dataset: DatasetLike, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset, dict):
        x = dataset.get("x")
        y_data = dataset.get("y")
        if x is None or y_data is None:
            raise ValueError("Dataset dictionary must include 'x' and 'y'.")
        return np.asarray(x), np.asarray(y_data)
    if y is None:
        raise ValueError("Both x and y arrays are required.")
    return np.asarray(dataset), np.asarray(y)


class BaseSurrogate:
    """Minimal base class for surrogate models."""

    def fit(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "BaseSurrogate":
        raise NotImplementedError

    def predict(self, x: np.ndarray, params: Optional[np.ndarray] = None, **kwargs: Any) -> SurrogatePrediction:
        raise NotImplementedError

    def update(self, dataset: DatasetLike, y: Optional[np.ndarray] = None, **kwargs: Any) -> "BaseSurrogate":
        return self.fit(dataset, y, **kwargs)

    def evaluate(self, dataset: DatasetLike, metrics: Optional[Iterable[Any]] = None) -> Dict[str, float]:
        from .. import metrics as metric_lib

        x, y = _extract_xy(dataset)
        pred = self.predict(x)
        if metrics is None:
            metrics = [metric_lib.rmse, metric_lib.nrmse]
        results: Dict[str, float] = {}
        for metric in metrics:
            name = getattr(metric, "__name__", "metric")
            if name in {"nll_gaussian", "crps_gaussian", "coverage_probability", "calibration_error"}:
                if pred.std is None:
                    continue
                results[name] = metric(pred.mean, pred.std, y)
            else:
                results[name] = metric(pred.mean, y)
        return results

    def save(self, path: str) -> None:
        np.savez(path, **{"class": self.__class__.__name__})

    @classmethod
    def load(cls, path: str) -> "BaseSurrogate":
        _ = np.load(path, allow_pickle=True)
        return cls()  # type: ignore[call-arg]
