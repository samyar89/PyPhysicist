"""Environment and force models for rocketry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from PyPhysicist.constants import GRAVITATIONAL_CONSTANT
from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


@dataclass(frozen=True)
class GravityModel:
    """Base gravity model."""

    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def acceleration(self, position: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformGravity(GravityModel):
    """Uniform gravity with constant magnitude."""

    gravity: Quantity | float = 9.80665
    direction: Sequence[float] = (0.0, -1.0, 0.0)
    assumptions: tuple[str, ...] = ("uniform gravity",)

    def acceleration(self, position: Any = None) -> Any:
        g_value, _ = coerce_value(self.gravity, "m/s^2", name="gravity")
        direction = np.asarray(self.direction, dtype=float)
        if np.allclose(direction, 0.0):
            raise ValueError("Gravity direction must be non-zero.")
        unit_dir = direction / _norm(direction)
        accel = g_value * unit_dir
        return wrap_quantity(accel, "m/s^2", self.gravity)


@dataclass(frozen=True)
class InverseSquareGravity(GravityModel):
    """Inverse-square gravity model for point masses."""

    central_mass: Quantity | float
    assumptions: tuple[str, ...] = ("point-mass gravity",)

    def acceleration(self, position: Any) -> Any:
        pos_value, _ = coerce_value(position, "m", name="position")
        r = _norm(pos_value)
        if r == 0:
            raise ValueError("Position must be non-zero for inverse-square gravity.")
        mu_value, _ = coerce_value(self.central_mass, "kg", name="central_mass")
        accel = -(GRAVITATIONAL_CONSTANT * mu_value / r**3) * pos_value
        return wrap_quantity(accel, "m/s^2", position, self.central_mass)


@dataclass(frozen=True)
class AtmosphereModel:
    """Base atmospheric density model."""

    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def density(self, altitude: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class ExponentialAtmosphere(AtmosphereModel):
    """Exponential atmosphere density profile."""

    sea_level_density: Quantity | float = 1.225
    scale_height: Quantity | float = 8500.0
    assumptions: tuple[str, ...] = (
        "isothermal atmosphere",
        "hydrostatic equilibrium",
    )

    def density(self, altitude: Any) -> Any:
        rho0_value, _ = coerce_value(self.sea_level_density, "kg/m^3", name="sea_level_density")
        h_value, _ = coerce_value(altitude, "m", name="altitude")
        h_scale_value, _ = coerce_value(self.scale_height, "m", name="scale_height")
        rho = rho0_value * np.exp(-h_value / h_scale_value)
        return wrap_quantity(rho, "kg/m^3", self.sea_level_density, altitude)


@dataclass(frozen=True)
class LayeredAtmosphere(AtmosphereModel):
    """Layered atmosphere with piecewise density values."""

    layers: Sequence[tuple[Quantity | float, Quantity | float]]
    assumptions: tuple[str, ...] = ("piecewise-constant density",)

    def density(self, altitude: Any) -> Any:
        altitude_value, _ = coerce_value(altitude, "m", name="altitude")
        for cutoff, density in self.layers:
            cutoff_value, _ = coerce_value(cutoff, "m", name="cutoff")
            if altitude_value <= cutoff_value:
                density_value, _ = coerce_value(density, "kg/m^3", name="density")
                return wrap_quantity(density_value, "kg/m^3", density, altitude)
        last_density = self.layers[-1][1]
        density_value, _ = coerce_value(last_density, "kg/m^3", name="density")
        return wrap_quantity(density_value, "kg/m^3", last_density, altitude)


@dataclass(frozen=True)
class DragModel:
    """Quadratic drag force model."""

    drag_coefficient: float = 0.0
    assumptions: tuple[str, ...] = ("quadratic drag",)

    def force(self, density: Any, velocity: Any, area: Any) -> Any:
        rho_value, _ = coerce_value(density, "kg/m^3", name="density")
        vel_value, _ = coerce_value(velocity, "m/s", name="velocity")
        area_value, _ = coerce_value(area, "m^2", name="area")
        speed = _norm(np.asarray(vel_value))
        drag_mag = 0.5 * rho_value * self.drag_coefficient * area_value * speed**2
        drag_vec = -drag_mag * (vel_value / speed) if speed != 0 else np.zeros_like(vel_value)
        return wrap_quantity(drag_vec, "N", density, velocity, area)


@dataclass(frozen=True)
class LiftModel:
    """Simplified lift force model."""

    lift_coefficient: float = 0.0
    lift_direction: Sequence[float] = (0.0, 1.0, 0.0)
    assumptions: tuple[str, ...] = ("quadratic lift",)

    def force(self, density: Any, velocity: Any, area: Any) -> Any:
        rho_value, _ = coerce_value(density, "kg/m^3", name="density")
        vel_value, _ = coerce_value(velocity, "m/s", name="velocity")
        area_value, _ = coerce_value(area, "m^2", name="area")
        speed = _norm(np.asarray(vel_value))
        lift_mag = 0.5 * rho_value * self.lift_coefficient * area_value * speed**2
        direction = np.asarray(self.lift_direction, dtype=float)
        if np.allclose(direction, 0.0):
            raise ValueError("Lift direction must be non-zero.")
        unit_dir = direction / _norm(direction)
        lift_vec = lift_mag * unit_dir
        return wrap_quantity(lift_vec, "N", density, velocity, area)


__all__ = [
    "GravityModel",
    "UniformGravity",
    "InverseSquareGravity",
    "AtmosphereModel",
    "ExponentialAtmosphere",
    "LayeredAtmosphere",
    "DragModel",
    "LiftModel",
]
