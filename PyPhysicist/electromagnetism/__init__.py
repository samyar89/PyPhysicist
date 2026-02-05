"""Electromagnetism subpackage."""

from .circuits import (
    I,
    R,
    V,
    current,
    resistance,
    resistance_parallel,
    resistance_series,
    voltage,
)
from .electrostatics import capacitance, coulomb_force, electric_field

__all__ = [
    "I",
    "R",
    "V",
    "current",
    "resistance",
    "resistance_parallel",
    "resistance_series",
    "voltage",
    "capacitance",
    "coulomb_force",
    "electric_field",
]
