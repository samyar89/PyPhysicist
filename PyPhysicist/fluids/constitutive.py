"""Constitutive relations and closures for fluid mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import FlowField


@dataclass
class NewtonianStress:
    """Newtonian stress tensor model."""

    dynamic_viscosity: float

    def __call__(self, velocity_gradient: np.ndarray, pressure: float) -> np.ndarray:
        strain_rate = 0.5 * (velocity_gradient + np.swapaxes(velocity_gradient, -1, -2))
        identity = np.eye(velocity_gradient.shape[-1])
        return -pressure * identity + 2.0 * self.dynamic_viscosity * strain_rate


@dataclass
class PowerLawViscosity:
    """Power-law non-Newtonian viscosity model."""

    consistency_index: float
    flow_behavior_index: float

    def __call__(self, shear_rate: np.ndarray) -> np.ndarray:
        return self.consistency_index * np.power(shear_rate, self.flow_behavior_index - 1.0)


@dataclass
class BinghamPlastic:
    """Bingham plastic viscosity model."""

    yield_stress: float
    plastic_viscosity: float

    def __call__(self, shear_rate: np.ndarray) -> np.ndarray:
        return self.plastic_viscosity + self.yield_stress / np.maximum(shear_rate, 1e-12)


@dataclass
class FourierHeatConduction:
    """Fourier heat conduction law."""

    conductivity: float

    def __call__(self, temperature_gradient: np.ndarray) -> np.ndarray:
        return -self.conductivity * temperature_gradient


@dataclass
class MixingLengthClosure:
    """Algebraic turbulence closure based on mixing-length."""

    mixing_length: Callable[[np.ndarray], np.ndarray]
    kappa: float = 0.41

    def __call__(self, velocity_gradient: np.ndarray) -> np.ndarray:
        shear_rate = np.linalg.norm(velocity_gradient, axis=(-2, -1))
        length = self.mixing_length(shear_rate)
        return (self.kappa * length) ** 2 * shear_rate


def eddy_viscosity_algebraic(flow: FlowField, coefficient: float) -> np.ndarray:
    """Simple algebraic eddy viscosity proportional to speed."""

    velocity = flow.evaluate_velocity(flow.grid)
    speed = np.linalg.norm(velocity, axis=-1)
    return coefficient * speed


__all__ = [
    "NewtonianStress",
    "PowerLawViscosity",
    "BinghamPlastic",
    "FourierHeatConduction",
    "MixingLengthClosure",
    "eddy_viscosity_algebraic",
]
