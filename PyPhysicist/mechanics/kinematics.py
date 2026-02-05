"""Kinematics formulas."""

import numpy as np


def velocity(distance: float, time: float):
    """Calculate velocity from distance and time.

    Dimensional safety is critical: ensure distance and time share compatible
    units (e.g., meters and seconds).
    """
    distance = np.asarray(distance)
    time = np.asarray(time)
    return distance / time


def centripetal_acceleration(speed: float, radius: float):
    """Calculate centripetal acceleration.

    Dimensional safety is critical: speed and radius must be in compatible
    units (e.g., m/s and m).
    """
    speed = np.asarray(speed)
    radius = np.asarray(radius)
    return (speed ** 2) / radius


Velocity = velocity

__all__ = ["velocity", "centripetal_acceleration", "Velocity"]
