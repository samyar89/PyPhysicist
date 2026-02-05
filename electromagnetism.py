"""Electromagnetism formulas."""

import numpy as np


def coulomb_force(q1: float, q2: float, r: float):
    """Calculate electrostatic force using Coulomb's law.

    Supports scalar or NumPy array-like inputs.

    Args:
        q1: Charge 1 in coulombs (C).
        q2: Charge 2 in coulombs (C).
        r: Separation distance in meters (m).

    Returns:
        Electrostatic force in newtons (N).
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    r = np.asarray(r)
    k = 8.9875517923 * (10 ** 9)
    return k * (q1 * q2) / (r ** 2)


def electric_field(force: float, charge: float):
    """Calculate electric field strength.

    Supports scalar or NumPy array-like inputs.

    Args:
        force: Force in newtons (N).
        charge: Charge in coulombs (C).

    Returns:
        Electric field strength in newtons per coulomb (N/C).
    """
    force = np.asarray(force)
    charge = np.asarray(charge)
    return force / charge


def capacitance(charge: float, voltage: float):
    """Calculate capacitance.

    Supports scalar or NumPy array-like inputs.

    Args:
        charge: Charge in coulombs (C).
        voltage: Voltage in volts (V).

    Returns:
        Capacitance in farads (F).
    """
    charge = np.asarray(charge)
    voltage = np.asarray(voltage)
    return charge / voltage


def resistance_series(*resistances: float):
    """Calculate equivalent resistance for series resistors.

    Supports scalar or NumPy array-like inputs.

    Args:
        *resistances: Individual resistances in ohms (Ω).

    Returns:
        Equivalent resistance in ohms (Ω).
    """
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for resistance in resistances:
        total = total + np.asarray(resistance)
    return total


def resistance_parallel(*resistances: float):
    """Calculate equivalent resistance for parallel resistors.

    Supports scalar or NumPy array-like inputs.

    Args:
        *resistances: Individual resistances in ohms (Ω).

    Returns:
        Equivalent resistance in ohms (Ω).
    """
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for resistance in resistances:
        total = total + (1 / np.asarray(resistance))
    return 1 / total
