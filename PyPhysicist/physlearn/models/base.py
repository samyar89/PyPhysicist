"""Base interfaces for surrogate models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SurrogatePrediction:
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseSurrogate:
    """Minimal base class for surrogate models."""

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseSurrogate":
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        raise NotImplementedError

    def update(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseSurrogate":
        return self.fit(x, y, **kwargs)
