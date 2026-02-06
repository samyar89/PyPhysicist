"""Synthetic datasets for PhysLearn benchmarks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def burgers_1d(n_samples: int = 128, viscosity: float = 0.01, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_samples)[:, None]
    params = rng.uniform(0.5, 1.5, size=(n_samples, 1))
    u = np.sin(2 * np.pi * x * params) * np.exp(-viscosity * x)
    return {"x": x, "y": u}


def heat_1d(n_samples: int = 128, conductivity: float = 0.5) -> Dict[str, np.ndarray]:
    x = np.linspace(0, 1, n_samples)
    u = np.exp(-conductivity * x)[:, None]
    return {"x": x[:, None], "y": u}


def shallow_water_toy(n_samples: int = 128, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_samples)
    h = 1 + 0.1 * np.sin(2 * np.pi * x)
    u = rng.normal(0.0, 0.05, size=n_samples)
    return {"x": np.stack([x, h], axis=1), "y": u[:, None]}


def low_high_fidelity_pair(dataset_fn, *args, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    low = dataset_fn(*args, **kwargs)
    high = dataset_fn(*args, **kwargs)
    high["y"] = high["y"] + 0.01 * np.sin(np.linspace(0, 2 * np.pi, high["y"].shape[0]))[:, None]
    return low, high
