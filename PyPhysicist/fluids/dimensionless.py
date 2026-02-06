"""Dimensionless analysis utilities for fluid mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from .core import FluidProperties, _quantity_value


@dataclass
class DimensionlessGroup:
    name: str
    value: float
    definition: str


def reynolds_number(properties: FluidProperties, velocity: float, length: float) -> DimensionlessGroup:
    rho = _quantity_value(properties.density)
    mu = _quantity_value(properties.dynamic_viscosity)
    value = rho * velocity * length / mu
    return DimensionlessGroup("Re", float(value), "ρ U L / μ")


def mach_number(velocity: float, speed_of_sound: float) -> DimensionlessGroup:
    return DimensionlessGroup("Ma", float(velocity / speed_of_sound), "U / a")


def prandtl_number(kinematic_viscosity: float, thermal_diffusivity: float) -> DimensionlessGroup:
    return DimensionlessGroup("Pr", float(kinematic_viscosity / thermal_diffusivity), "ν / α")


def nusselt_number(convective_htc: float, length: float, conductivity: float) -> DimensionlessGroup:
    return DimensionlessGroup("Nu", float(convective_htc * length / conductivity), "h L / k")


def froude_number(velocity: float, length: float, gravity: float = 9.81) -> DimensionlessGroup:
    return DimensionlessGroup("Fr", float(velocity / np.sqrt(gravity * length)), "U / sqrt(gL)")


def strouhal_number(frequency: float, length: float, velocity: float) -> DimensionlessGroup:
    return DimensionlessGroup("St", float(frequency * length / velocity), "f L / U")


def similarity_parameters(groups: Iterable[DimensionlessGroup]) -> Dict[str, float]:
    return {group.name: group.value for group in groups}


def verify_similarity(groups_a: Iterable[DimensionlessGroup], groups_b: Iterable[DimensionlessGroup], tol: float = 1e-6) -> bool:
    ref_a = similarity_parameters(groups_a)
    ref_b = similarity_parameters(groups_b)
    for key, value in ref_a.items():
        if key not in ref_b:
            return False
        if abs(value - ref_b[key]) > tol:
            return False
    return True


def nondimensionalize_equations(variables: Dict[str, float], scales: Dict[str, float]) -> Dict[str, float]:
    """Simple nondimensionalization map using provided variable scales."""

    return {name: value / scales[name] for name, value in variables.items() if name in scales}


__all__ = [
    "DimensionlessGroup",
    "reynolds_number",
    "mach_number",
    "prandtl_number",
    "nusselt_number",
    "froude_number",
    "strouhal_number",
    "similarity_parameters",
    "verify_similarity",
    "nondimensionalize_equations",
]
