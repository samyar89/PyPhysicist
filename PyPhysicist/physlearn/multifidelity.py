"""Multi-fidelity training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .models.base import BaseSurrogate, DatasetLike, _extract_xy


@dataclass
class MultiFidelityTrainer:
    """Pretrain low-fidelity then fine-tune on high-fidelity."""

    surrogate: BaseSurrogate

    def pretrain(self, low_fidelity: DatasetLike, **kwargs: Any) -> BaseSurrogate:
        self.surrogate.fit(low_fidelity, **kwargs)
        return self.surrogate

    def finetune(self, high_fidelity: DatasetLike, **kwargs: Any) -> BaseSurrogate:
        self.surrogate.fit(high_fidelity, **kwargs)
        return self.surrogate


def discrepancy_dataset(low: DatasetLike, high: DatasetLike) -> dict:
    x_low, y_low = _extract_xy(low)
    x_high, y_high = _extract_xy(high)
    if not np.allclose(x_low, x_high):
        raise ValueError("Low- and high-fidelity datasets must share the same inputs.")
    return {"x": x_low, "y": y_high - y_low}
