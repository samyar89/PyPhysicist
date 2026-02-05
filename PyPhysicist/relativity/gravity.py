"""Relativistic gravity formulas."""

import numpy as np

from ..constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT


def schwarzschild_radius(mass: float, g: float = GRAVITATIONAL_CONSTANT, c: float = SPEED_OF_LIGHT):
    """Calculate the Schwarzschild radius for a given mass.

    Dimensional safety is critical: mass, g, and c must be in compatible units.
    """
    mass = np.asarray(mass)
    g = np.asarray(g)
    c = np.asarray(c)
    return (2 * g * mass) / (c ** 2)


Schwarzschild_radius = schwarzschild_radius

__all__ = ["schwarzschild_radius", "Schwarzschild_radius"]
