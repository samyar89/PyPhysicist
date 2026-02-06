"""Special relativity formulas."""

import numpy as np

from ..constants import SPEED_OF_LIGHT
from ..units import coerce_value, wrap_quantity


def time_dilation(proper_time: float, velocity: float, c: float = SPEED_OF_LIGHT):
    """Calculate time dilation.

    Dimensional safety is critical: velocity and c must share units.
    """
    proper_time_value, _ = coerce_value(proper_time, "s", name="proper_time")
    velocity_value, _ = coerce_value(velocity, "m/s", name="velocity")
    c_value, _ = coerce_value(c, "m/s", name="c")
    gamma = 1 / np.sqrt(1 - (velocity_value ** 2) / (c_value ** 2))
    result = proper_time_value * gamma
    return wrap_quantity(result, "s", proper_time, velocity, c)


def length_contraction(proper_length: float, velocity: float, c: float = SPEED_OF_LIGHT):
    """Calculate length contraction.

    Dimensional safety is critical: velocity and c must share units.
    """
    proper_length_value, _ = coerce_value(proper_length, "m", name="proper_length")
    velocity_value, _ = coerce_value(velocity, "m/s", name="velocity")
    c_value, _ = coerce_value(c, "m/s", name="c")
    gamma = 1 / np.sqrt(1 - (velocity_value ** 2) / (c_value ** 2))
    result = proper_length_value / gamma
    return wrap_quantity(result, "m", proper_length, velocity, c)


def relativistic_energy(mass: float, c: float = SPEED_OF_LIGHT):
    """Calculate rest energy.

    Dimensional safety is critical: use kg and m/s or quantities with units.
    """
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    c_value, _ = coerce_value(c, "m/s", name="c")
    result = mass_value * (c_value ** 2)
    return wrap_quantity(result, "J", mass, c)


__all__ = ["time_dilation", "length_contraction", "relativistic_energy"]
