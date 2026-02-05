"""Electromagnetism formulas."""


def coulomb_force(q1: float, q2: float, r: float):
    """
    q1: C
    q2: C
    r: m

    In this case, the electrostatic force is obtained in Newtons.
    """
    k = 8.9875517923 * (10 ** 9)
    return k * (q1 * q2) / (r ** 2)


def electric_field(force: float, charge: float):
    """
    force: N
    charge: C

    In this case, the electric field is obtained in N/C.
    """
    return force / charge


def capacitance(charge: float, voltage: float):
    """
    charge: C
    voltage: V

    In this case, the capacitance is obtained in Farads.
    """
    return charge / voltage


def resistance_series(*resistances: float):
    """
    resistances: Ω

    In this case, the equivalent resistance is obtained in Ohms.
    """
    return sum(resistances)


def resistance_parallel(*resistances: float):
    """
    resistances: Ω

    In this case, the equivalent resistance is obtained in Ohms.
    """
    total = 0.0
    for resistance in resistances:
        total += 1 / resistance
    return 1 / total
