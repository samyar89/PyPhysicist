"""Optics formulas."""

import numpy as np


def refractive_index(speed_of_light: float, medium_speed: float):
    """Calculate refractive index."""
    speed_of_light = np.asarray(speed_of_light)
    medium_speed = np.asarray(medium_speed)
    return speed_of_light / medium_speed


__all__ = ["refractive_index"]
