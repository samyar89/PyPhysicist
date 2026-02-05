"""Electrostatics formulas."""

import numpy as np

from ..constants import COULOMB_CONSTANT


def coulomb_force(charge1: float, charge2: float, distance: float):
    """Calculate electrostatic force using Coulomb's law.

    Dimensional safety is critical: ensure charges are in coulombs (C) and
    distance is in meters (m).
    """
    charge1 = np.asarray(charge1)
    charge2 = np.asarray(charge2)
    distance = np.asarray(distance)
    return COULOMB_CONSTANT * (charge1 * charge2) / (distance ** 2)


def electric_field(force_value: float, charge: float):
    """Calculate electric field strength."""
    force_value = np.asarray(force_value)
    charge = np.asarray(charge)
    return force_value / charge


def capacitance(charge: float, voltage: float):
    """Calculate capacitance."""
    charge = np.asarray(charge)
    voltage = np.asarray(voltage)
    return charge / voltage


__all__ = ["coulomb_force", "electric_field", "capacitance"]
