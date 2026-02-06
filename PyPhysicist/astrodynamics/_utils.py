"""Shared helpers for astrodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from PyPhysicist.units import Quantity, UnitError
from PyPhysicist.units.conversion import coerce_value, parse_unit


@dataclass(frozen=True)
class VectorQuantity:
    """Container for a vector with an associated unit."""

    value: np.ndarray
    unit: str


def _ensure_quantity(value, expected_unit: str, *, name: str) -> Tuple[np.ndarray, str]:
    if not isinstance(value, Quantity):
        raise UnitError(
            f"{name} must be provided as a Quantity with units of {expected_unit}."
        )
    numeric, unit = coerce_value(value, expected_unit, name=name)
    return np.asarray(numeric, dtype=float), unit or expected_unit


def require_vector_quantity(
    value, expected_unit: str, *, name: str, shape: Iterable[int] = (3,)
) -> VectorQuantity:
    """Ensure an input is a vector Quantity with the expected unit."""
    numeric, unit = _ensure_quantity(value, expected_unit, name=name)
    if numeric.shape != tuple(shape):
        raise ValueError(
            f"{name} must have shape {shape}, got {numeric.shape} instead."
        )
    return VectorQuantity(value=numeric, unit=unit)


def require_scalar_quantity(value, expected_unit: str, *, name: str) -> Quantity:
    """Ensure an input is a scalar Quantity with the expected unit."""
    numeric, unit = _ensure_quantity(value, expected_unit, name=name)
    if np.asarray(numeric).shape not in ((), (1,)):
        raise ValueError(f"{name} must be a scalar quantity.")
    return Quantity(float(np.asarray(numeric)), unit)


def vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def unit_dimensions(unit: str) -> dict:
    return parse_unit(unit).dims


__all__ = [
    "VectorQuantity",
    "require_vector_quantity",
    "require_scalar_quantity",
    "vector_norm",
    "unit_dimensions",
]
