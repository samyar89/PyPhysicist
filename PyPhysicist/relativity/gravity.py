"""Relativistic gravity formulas."""

from ..constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT
from ..units import coerce_value, wrap_quantity


def schwarzschild_radius(mass: float, g: float = GRAVITATIONAL_CONSTANT, c: float = SPEED_OF_LIGHT):
    """Calculate the Schwarzschild radius for a given mass.

    Dimensional safety is critical: mass, g, and c must be in compatible units.
    """
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    g_value, _ = coerce_value(g, "m^3/kg*s^2", name="g")
    c_value, _ = coerce_value(c, "m/s", name="c")
    result = (2 * g_value * mass_value) / (c_value ** 2)
    return wrap_quantity(result, "m", mass, g, c)


Schwarzschild_radius = schwarzschild_radius

__all__ = ["schwarzschild_radius", "Schwarzschild_radius"]
