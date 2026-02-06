"""Integral theorems and conservation laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import ControlVolume, FlowField, FluidProperties, _quantity_value


@dataclass
class TheoremResult:
    name: str
    value: float | np.ndarray
    symbolic: str


def reynolds_transport(
    control_volume: ControlVolume,
    flow: FlowField,
    properties: FluidProperties,
    property_density: Callable[[np.ndarray], float],
    time: float | None = None,
) -> TheoremResult:
    """Reynolds Transport Theorem for a conserved property."""

    flux = 0.0
    for surface in control_volume.surfaces:
        velocity = flow.evaluate_velocity(surface.centroid, time)
        phi = property_density(surface.centroid)
        flux += _quantity_value(properties.density) * phi * float(np.dot(velocity, surface.normal)) * surface.area
    return TheoremResult("ReynoldsTransport", float(flux), "d/dt ∫_CV ρφ dV + ∮_CS ρφ u·n dA")


def bernoulli(flow: FlowField, properties: FluidProperties, elevation: float = 0.0) -> TheoremResult:
    """Bernoulli equation along a streamline."""

    velocity = flow.evaluate_velocity(flow.grid)
    pressure = flow.evaluate_pressure(flow.grid)
    rho = _quantity_value(properties.density)
    speed_sq = np.sum(np.asarray(velocity) ** 2, axis=-1)
    value = pressure / rho + 0.5 * speed_sq + 9.81 * elevation
    return TheoremResult("Bernoulli", value, "p/ρ + u^2/2 + g z")


def kelvin_circulation(flow: FlowField, loop_points: np.ndarray) -> TheoremResult:
    """Kelvin circulation theorem evaluation."""

    velocity = flow.evaluate_velocity(loop_points)
    tangent = np.gradient(loop_points, axis=0)
    circulation = np.sum(np.einsum("ij,ij->i", velocity, tangent))
    return TheoremResult("KelvinCirculation", float(circulation), "∮ u·dl")


def momentum_balance(control_volume: ControlVolume, flow: FlowField, properties: FluidProperties) -> TheoremResult:
    flux = control_volume.momentum_flux(flow, properties)
    return TheoremResult("MomentumBalance", flux, "∮ (ρuu + pI)·n dA")


def energy_balance(
    control_volume: ControlVolume,
    flow: FlowField,
    properties: FluidProperties,
    specific_energy: Callable[[np.ndarray], float],
) -> TheoremResult:
    flux = control_volume.energy_flux(flow, properties, specific_energy)
    return TheoremResult("EnergyBalance", flux, "∮ ρe u·n dA")


__all__ = [
    "TheoremResult",
    "reynolds_transport",
    "bernoulli",
    "kelvin_circulation",
    "momentum_balance",
    "energy_balance",
]
