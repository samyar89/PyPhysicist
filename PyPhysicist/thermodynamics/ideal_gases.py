"""Ideal gas relationships."""

import numpy as np

from ..constants import IDEAL_GAS_CONSTANT


def ideal_gas_pressure(n: float, r: float = IDEAL_GAS_CONSTANT, t: float = None, v: float = None):
    """Calculate pressure using the ideal gas law.

    Dimensional safety is critical: ensure inputs are in SI units or use
    quantities with explicit units.

    Args:
        n: Amount of substance in moles (mol).
        r: Ideal gas constant in J/(mol·K). Defaults to SI R.
        t: Temperature in kelvin (K).
        v: Volume in cubic meters (m^3).
    """
    if t is None or v is None:
        raise ValueError("Both temperature 't' and volume 'v' must be provided.")
    n = np.asarray(n)
    r = np.asarray(r)
    t = np.asarray(t)
    v = np.asarray(v)
    return (n * r * t) / v


__all__ = ["ideal_gas_pressure"]
