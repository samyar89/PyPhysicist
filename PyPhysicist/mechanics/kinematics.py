"""Kinematics formulas."""

from ..units import coerce_value, wrap_quantity


def velocity(distance: float, time: float):
    """Calculate velocity from distance and time.

    Dimensional safety is critical: ensure distance and time share compatible
    units (e.g., meters and seconds).
    """
    distance_value, _ = coerce_value(distance, "m", name="distance")
    time_value, _ = coerce_value(time, "s", name="time")
    result = distance_value / time_value
    return wrap_quantity(result, "m/s", distance, time)


def centripetal_acceleration(speed: float, radius: float):
    """Calculate centripetal acceleration.

    Dimensional safety is critical: speed and radius must be in compatible
    units (e.g., m/s and m).
    """
    speed_value, _ = coerce_value(speed, "m/s", name="speed")
    radius_value, _ = coerce_value(radius, "m", name="radius")
    result = (speed_value ** 2) / radius_value
    return wrap_quantity(result, "m/s^2", speed, radius)


Velocity = velocity

__all__ = ["velocity", "centripetal_acceleration", "Velocity"]
