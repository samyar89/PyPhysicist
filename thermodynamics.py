"""Thermodynamics formulas."""


def ideal_gas_pressure(n: float, r: float, t: float, v: float):
    """Calculate pressure using the ideal gas law.

    Args:
        n: Amount of substance in moles (mol).
        r: Ideal gas constant in J/(mol·K).
        t: Temperature in kelvin (K).
        v: Volume in cubic meters (m³).

    Returns:
        Pressure in pascals (Pa).
    """
    return (n * r * t) / v


def heat_capacity(heat: float, mass: float, delta_t: float):
    """Calculate specific heat capacity.

    Args:
        heat: Heat energy in joules (J).
        mass: Mass in kilograms (kg).
        delta_t: Temperature change in kelvin (K).

    Returns:
        Specific heat capacity in J/(kg·K).
    """
    return heat / (mass * delta_t)


def entropy_change(heat: float, temperature: float):
    """Calculate entropy change.

    Args:
        heat: Heat energy in joules (J).
        temperature: Absolute temperature in kelvin (K).

    Returns:
        Entropy change in joules per kelvin (J/K).
    """
    return heat / temperature
