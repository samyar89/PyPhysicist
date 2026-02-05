"""Special relativity formulas."""

import numpy as np

from ..constants import SPEED_OF_LIGHT


def time_dilation(proper_time: float, velocity: float, c: float = SPEED_OF_LIGHT):
    """Calculate time dilation.

    Dimensional safety is critical: velocity and c must share units.
    """
    proper_time = np.asarray(proper_time)
    velocity = np.asarray(velocity)
    c = np.asarray(c)
    gamma = 1 / np.sqrt(1 - (velocity ** 2) / (c ** 2))
    return proper_time * gamma


def length_contraction(proper_length: float, velocity: float, c: float = SPEED_OF_LIGHT):
    """Calculate length contraction.

    Dimensional safety is critical: velocity and c must share units.
    """
    proper_length = np.asarray(proper_length)
    velocity = np.asarray(velocity)
    c = np.asarray(c)
    gamma = 1 / np.sqrt(1 - (velocity ** 2) / (c ** 2))
    return proper_length / gamma


def relativistic_energy(mass: float, c: float = SPEED_OF_LIGHT):
    """Calculate rest energy.

    Dimensional safety is critical: use kg and m/s or quantities with units.
    """
    mass = np.asarray(mass)
    c = np.asarray(c)
    return mass * (c ** 2)


__all__ = ["time_dilation", "length_contraction", "relativistic_energy"]
