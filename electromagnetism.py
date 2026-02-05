"""Electromagnetism formulas."""


def coulomb_force(q1: float, q2: float, r: float):
    """Calculate electrostatic force using Coulomb's law.

    Args:
        q1: Charge 1 in coulombs (C).
        q2: Charge 2 in coulombs (C).
        r: Separation distance in meters (m).

    Returns:
        Electrostatic force in newtons (N).
    """
    k = 8.9875517923 * (10 ** 9)
    return k * (q1 * q2) / (r ** 2)


def electric_field(force: float, charge: float):
    """Calculate electric field strength.

    Args:
        force: Force in newtons (N).
        charge: Charge in coulombs (C).

    Returns:
        Electric field strength in newtons per coulomb (N/C).
    """
    return force / charge


def capacitance(charge: float, voltage: float):
    """Calculate capacitance.

    Args:
        charge: Charge in coulombs (C).
        voltage: Voltage in volts (V).

    Returns:
        Capacitance in farads (F).
    """
    return charge / voltage


def resistance_series(*resistances: float):
    """Calculate equivalent resistance for series resistors.

    Args:
        *resistances: Individual resistances in ohms (Ω).

    Returns:
        Equivalent resistance in ohms (Ω).
    """
    return sum(resistances)


def resistance_parallel(*resistances: float):
    """Calculate equivalent resistance for parallel resistors.

    Args:
        *resistances: Individual resistances in ohms (Ω).

    Returns:
        Equivalent resistance in ohms (Ω).
    """
    total = 0.0
    for resistance in resistances:
        total += 1 / resistance
    return 1 / total
