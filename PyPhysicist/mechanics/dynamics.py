"""Dynamics formulas."""

import numpy as np


def force(mass: float, acceleration: float):
    """Calculate force using Newton's second law.

    Dimensional safety is critical: mass and acceleration must be in
    compatible units (e.g., kg and m/s^2).
    """
    mass = np.asarray(mass)
    acceleration = np.asarray(acceleration)
    return mass * acceleration


def newton_second_law(mass: float, acceleration: float):
    """Alias for :func:`force`."""
    return force(mass=mass, acceleration=acceleration)


def momentum(mass: float, velocity: float):
    """Calculate linear momentum.

    Dimensional safety is critical: mass and velocity must be in compatible
    units (e.g., kg and m/s).
    """
    mass = np.asarray(mass)
    velocity = np.asarray(velocity)
    return mass * velocity


def centripetal_force(mass: float, speed: float, radius: float):
    """Calculate centripetal force for circular motion."""
    mass = np.asarray(mass)
    speed = np.asarray(speed)
    radius = np.asarray(radius)
    return mass * (speed ** 2) / radius


def weight(mass: float, gravity: float):
    """Calculate weight from mass and gravitational acceleration."""
    mass = np.asarray(mass)
    gravity = np.asarray(gravity)
    return mass * gravity


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
