"""Relativity formulas."""

import numpy as np


def time_dilation(proper_time: float, velocity: float, c: float):
    """Calculate time dilation.

    Supports scalar or NumPy array-like inputs.

    Args:
        proper_time: Proper time in seconds (s).
        velocity: Relative velocity in meters per second (m/s).
        c: Speed of light in meters per second (m/s).

    Returns:
        Dilated time in seconds (s).
    """
    proper_time = np.asarray(proper_time)
    velocity = np.asarray(velocity)
    c = np.asarray(c)
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_time * gamma


def length_contraction(proper_length: float, velocity: float, c: float):
    """Calculate length contraction.

    Supports scalar or NumPy array-like inputs.

    Args:
        proper_length: Proper length in meters (m).
        velocity: Relative velocity in meters per second (m/s).
        c: Speed of light in meters per second (m/s).

    Returns:
        Contracted length in meters (m).
    """
    proper_length = np.asarray(proper_length)
    velocity = np.asarray(velocity)
    c = np.asarray(c)
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_length / gamma


def relativistic_energy(mass: float, c: float):
    """Calculate rest energy.

    Supports scalar or NumPy array-like inputs.

    Args:
        mass: Mass in kilograms (kg).
        c: Speed of light in meters per second (m/s).

    Returns:
        Rest energy in joules (J).
    """
    mass = np.asarray(mass)
    c = np.asarray(c)
    return mass * (c ** 2)
