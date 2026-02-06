"""Optics formulas."""

from ..units import coerce_value, wrap_quantity


def refractive_index(speed_of_light: float, medium_speed: float):
    """Calculate refractive index."""
    speed_of_light_value, _ = coerce_value(speed_of_light, "m/s", name="speed_of_light")
    medium_speed_value, _ = coerce_value(medium_speed, "m/s", name="medium_speed")
    result = speed_of_light_value / medium_speed_value
    return wrap_quantity(result, "1", speed_of_light, medium_speed)


__all__ = ["refractive_index"]
