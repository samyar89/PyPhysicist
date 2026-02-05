"""Thermodynamics formulas."""

import numpy as np


def ideal_gas_pressure(n: float, r: float, t: float, v: float):
    """Calculate pressure using the ideal gas law.

    Supports scalar or NumPy array-like inputs.

    Args:
        n: Amount of substance in moles (mol).
        r: Ideal gas constant in J/(mol·K).
        t: Temperature in kelvin (K).
        v: Volume in cubic meters (m³).

    Returns:
        Pressure in pascals (Pa).
    """
    n = np.asarray(n)
    r = np.asarray(r)
    t = np.asarray(t)
    v = np.asarray(v)
    return (n * r * t) / v


def heat_capacity(heat: float, mass: float, delta_t: float):
    """Calculate specific heat capacity.

    Supports scalar or NumPy array-like inputs.

    Args:
        heat: Heat energy in joules (J).
        mass: Mass in kilograms (kg).
        delta_t: Temperature change in kelvin (K).

    Returns:
        Specific heat capacity in J/(kg·K).
    """
    heat = np.asarray(heat)
    mass = np.asarray(mass)
    delta_t = np.asarray(delta_t)
    return heat / (mass * delta_t)


def entropy_change(heat: float, temperature: float):
    """Calculate entropy change.

    Supports scalar or NumPy array-like inputs.

    Args:
        heat: Heat energy in joules (J).
        temperature: Absolute temperature in kelvin (K).

    Returns:
        Entropy change in joules per kelvin (J/K).
    """
    heat = np.asarray(heat)
    temperature = np.asarray(temperature)
    return heat / temperature
