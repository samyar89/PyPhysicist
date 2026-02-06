"""Dynamics formulas."""

from ..units import coerce_value, wrap_quantity


def force(mass: float, acceleration: float):
    """Calculate force using Newton's second law.

    Dimensional safety is critical: mass and acceleration must be in
    compatible units (e.g., kg and m/s^2).
    """
    mass_value, mass_unit = coerce_value(mass, "kg", name="mass")
    acc_value, acc_unit = coerce_value(acceleration, "m/s^2", name="acceleration")
    result = mass_value * acc_value
    return wrap_quantity(result, "N", mass, acceleration)


def newton_second_law(mass: float, acceleration: float):
    """Alias for :func:`force`."""
    return force(mass=mass, acceleration=acceleration)


def momentum(mass: float, velocity: float):
    """Calculate linear momentum.

    Dimensional safety is critical: mass and velocity must be in compatible
    units (e.g., kg and m/s).
    """
    mass_value, mass_unit = coerce_value(mass, "kg", name="mass")
    velocity_value, velocity_unit = coerce_value(velocity, "m/s", name="velocity")
    result = mass_value * velocity_value
    return wrap_quantity(result, "kg*m/s", mass, velocity)


def centripetal_force(mass: float, speed: float, radius: float):
    """Calculate centripetal force for circular motion."""
    mass_value, mass_unit = coerce_value(mass, "kg", name="mass")
    speed_value, speed_unit = coerce_value(speed, "m/s", name="speed")
    radius_value, radius_unit = coerce_value(radius, "m", name="radius")
    result = mass_value * (speed_value ** 2) / radius_value
    return wrap_quantity(result, "N", mass, speed, radius)


def weight(mass: float, gravity: float):
    """Calculate weight from mass and gravitational acceleration."""
    mass_value, mass_unit = coerce_value(mass, "kg", name="mass")
    gravity_value, gravity_unit = coerce_value(gravity, "m/s^2", name="gravity")
    result = mass_value * gravity_value
    return wrap_quantity(result, "N", mass, gravity)


F = force
Weight = weight

__all__ = [
    "force",
    "newton_second_law",
    "momentum",
    "centripetal_force",
    "weight",
    "F",
    "Weight",
]
