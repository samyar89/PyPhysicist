"""Thermodynamics subpackage."""

from .heat import entropy_change, heat_capacity
from .ideal_gases import ideal_gas_pressure

__all__ = ["entropy_change", "heat_capacity", "ideal_gas_pressure"]
