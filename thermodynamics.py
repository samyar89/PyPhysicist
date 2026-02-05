"""Thermodynamics formulas."""


def ideal_gas_pressure(n: float, r: float, t: float, v: float):
    """
    n: mol
    r: J/(mol·K)
    t: K
    v: m^3

    In this case, the pressure is obtained in Pascals.
    """
    return (n * r * t) / v


def heat_capacity(heat: float, mass: float, delta_t: float):
    """
    heat: J
    mass: kg
    delta_t: K

    In this case, the specific heat capacity is obtained in J/(kg·K).
    """
    return heat / (mass * delta_t)


def entropy_change(heat: float, temperature: float):
    """
    heat: J
    temperature: K

    In this case, the entropy change is obtained in J/K.
    """
    return heat / temperature
