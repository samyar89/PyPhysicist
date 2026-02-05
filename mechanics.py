"""Classical mechanics formulas."""

import numpy as np


def momentum(mass: float, velocity: float):
    """Calculate linear momentum.

    Supports scalar or NumPy array-like inputs.

    Args:
        mass: Mass in kilograms (kg).
        velocity: Velocity in meters per second (m/s).

    Returns:
        Momentum in kilogram meters per second (kg·m/s).
    """
    mass = np.asarray(mass)
    velocity = np.asarray(velocity)
    return mass * velocity


def newton_second_law(mass: float, acceleration: float):
    """Calculate force using Newton's second law.

    Supports scalar or NumPy array-like inputs.

    Args:
        mass: Mass in kilograms (kg).
        acceleration: Acceleration in meters per second squared (m/s²).

    Returns:
        Force in newtons (N).
    """
    mass = np.asarray(mass)
    acceleration = np.asarray(acceleration)
    return mass * acceleration


def spring_potential_energy(k: float, x: float):
    """Calculate potential energy stored in a spring.

    Supports scalar or NumPy array-like inputs.

    Args:
        k: Spring constant in newtons per meter (N/m).
        x: Displacement from equilibrium in meters (m).

    Returns:
        Spring potential energy in joules (J).
    """
    k = np.asarray(k)
    x = np.asarray(x)
    return 0.5 * k * (x ** 2)


def centripetal_acceleration(velocity: float, radius: float):
    """Calculate centripetal acceleration.

    Supports scalar or NumPy array-like inputs.

    Args:
        velocity: Tangential speed in meters per second (m/s).
        radius: Radius of circular motion in meters (m).

    Returns:
        Centripetal acceleration in meters per second squared (m/s²).
    """
    velocity = np.asarray(velocity)
    radius = np.asarray(radius)
    return (velocity ** 2) / radius


def centripetal_force(mass: float, velocity: float, radius: float):
    """Calculate centripetal force for circular motion.

    Supports scalar or NumPy array-like inputs.

    Args:
        mass: Mass in kilograms (kg).
        velocity: Tangential speed in meters per second (m/s).
        radius: Radius of circular motion in meters (m).

    Returns:
        Centripetal force in newtons (N).
    """
    mass = np.asarray(mass)
    velocity = np.asarray(velocity)
    radius = np.asarray(radius)
    return mass * (velocity ** 2) / radius
