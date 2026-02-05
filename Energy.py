import numpy as np


def KINETIC_ENERGY(Mass: float, Velocity: float):
    """Calculate kinetic energy.

    Supports scalar or NumPy array-like inputs.

    Args:
        Mass: Mass in kilograms (kg).
        Velocity: Velocity in meters per second (m/s).

    Returns:
        Kinetic energy in joules (J).
    """
    Mass = np.asarray(Mass)
    Velocity = np.asarray(Velocity)
    return 0.5 * Mass * (Velocity ** 2)


def GRAVITATIONAL_POTENTIAL_ENERGY(
    Mass: float,
    Gravitational_acceleration: float,
    Height: float,
):
    """Calculate gravitational potential energy.

    Supports scalar or NumPy array-like inputs.

    Args:
        Mass: Mass in kilograms (kg).
        Gravitational_acceleration: Gravitational acceleration in meters per
            second squared (m/s²).
        Height: Height in meters (m).

    Returns:
        Gravitational potential energy in joules (J).
    """
    Mass = np.asarray(Mass)
    Gravitational_acceleration = np.asarray(Gravitational_acceleration)
    Height = np.asarray(Height)
    return Mass * Gravitational_acceleration * Height


def MECHANICAL_ENERGY(Kinetic_Energy: float, Gravitational_Potential_Energy: float):
    """Calculate total mechanical energy.

    Supports scalar or NumPy array-like inputs.

    Args:
        Kinetic_Energy: Kinetic energy in joules (J).
        Gravitational_Potential_Energy: Gravitational potential energy in joules (J).

    Returns:
        Total mechanical energy in joules (J).
    """
    Kinetic_Energy = np.asarray(Kinetic_Energy)
    Gravitational_Potential_Energy = np.asarray(Gravitational_Potential_Energy)
    return Kinetic_Energy + Gravitational_Potential_Energy


def ELASTIC_POTENTIAL_ENERGY(k: float, x: float):
    """Calculate elastic potential energy of a spring.

    Supports scalar or NumPy array-like inputs.

    Args:
        k: Spring constant in newtons per meter (N/m).
        x: Displacement from equilibrium in meters (m).

    Returns:
        Elastic potential energy in joules (J).
    """
    k = np.asarray(k)
    x = np.asarray(x)
    return 0.5 * k * (x ** 2)
