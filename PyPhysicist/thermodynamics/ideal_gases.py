"""Ideal gas relationships."""

from ..constants import IDEAL_GAS_CONSTANT
from ..units import coerce_value, wrap_quantity


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
    n_value, _ = coerce_value(n, "mol", name="n")
    r_value, _ = coerce_value(r, "J/mol*K", name="r")
    t_value, _ = coerce_value(t, "K", name="t")
    v_value, _ = coerce_value(v, "m^3", name="v")
    result = (n_value * r_value * t_value) / v_value
    return wrap_quantity(result, "Pa", n, r, t, v)


__all__ = ["ideal_gas_pressure"]
