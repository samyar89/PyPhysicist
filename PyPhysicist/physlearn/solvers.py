"""Lightweight physics solver adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class SolverOutput:
    field: np.ndarray
    metadata: Optional[Dict[str, float]] = None


@dataclass
class SolverAdapter:
    """Callable wrapper for physics solvers."""

    solver_fn: Callable[..., np.ndarray]

    def __call__(self, *args, **kwargs) -> SolverOutput:
        result = self.solver_fn(*args, **kwargs)
        return SolverOutput(field=np.asarray(result), metadata={"backend": "numpy"})


def heat_equation_fd(
    initial: np.ndarray,
    alpha: float,
    dx: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """Basic 1D heat equation finite-difference solver."""

    u = np.asarray(initial, dtype=float).copy()
    r = alpha * dt / (dx**2)
    for _ in range(steps):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u
