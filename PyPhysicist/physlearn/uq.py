"""Uncertainty quantification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .utils import optional_import, torch_available


@dataclass
class EnsembleUQ:
    members: List[Any]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = [member.predict(x).mean for member in self.members]
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0), stacked.var(axis=0)


def mc_dropout_predict(model: Any, x: np.ndarray, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if not torch_available():
        raise ImportError("PyTorch is required for MC-dropout.")
    torch = optional_import("torch", extra="torch")
    model.train()
    samples = []
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(x_tensor).cpu().numpy())
    stacked = np.stack(samples, axis=0)
    return stacked.mean(axis=0), stacked.var(axis=0)


def gpytorch_available() -> bool:
    try:
        optional_import("gpytorch", extra="gpytorch")
        return True
    except ImportError:
        return False
