"""Electrostatics formulas."""

from ..constants import COULOMB_CONSTANT
from ..units import coerce_value, wrap_quantity


def coulomb_force(charge1: float, charge2: float, distance: float):
    """Calculate electrostatic force using Coulomb's law.

    Dimensional safety is critical: ensure charges are in coulombs (C) and
    distance is in meters (m).
    """
    charge1_value, _ = coerce_value(charge1, "C", name="charge1")
    charge2_value, _ = coerce_value(charge2, "C", name="charge2")
    distance_value, _ = coerce_value(distance, "m", name="distance")
    result = COULOMB_CONSTANT * (charge1_value * charge2_value) / (distance_value ** 2)
    return wrap_quantity(result, "N", charge1, charge2, distance)


def electric_field(force_value: float, charge: float):
    """Calculate electric field strength."""
    force_value_value, _ = coerce_value(force_value, "N", name="force")
    charge_value, _ = coerce_value(charge, "C", name="charge")
    result = force_value_value / charge_value
    return wrap_quantity(result, "N/C", force_value, charge)


def capacitance(charge: float, voltage: float):
    """Calculate capacitance."""
    charge_value, _ = coerce_value(charge, "C", name="charge")
    voltage_value, _ = coerce_value(voltage, "V", name="voltage")
    result = charge_value / voltage_value
    return wrap_quantity(result, "F", charge, voltage)


__all__ = ["coulomb_force", "electric_field", "capacitance"]
