def KINETIC_ENERGY(Mass: float, Velocity: float):
    """Calculate kinetic energy.

    Args:
        Mass: Mass in kilograms (kg).
        Velocity: Velocity in meters per second (m/s).

    Returns:
        Kinetic energy in joules (J).
    """
    return 0.5 * Mass * (Velocity ** 2)


def GRAVITATIONAL_POTENTIAL_ENERGY(
    Mass: float,
    Gravitational_acceleration: float,
    Height: float,
):
    """Calculate gravitational potential energy.

    Args:
        Mass: Mass in kilograms (kg).
        Gravitational_acceleration: Gravitational acceleration in meters per
            second squared (m/s²).
        Height: Height in meters (m).

    Returns:
        Gravitational potential energy in joules (J).
    """
    return Mass * Gravitational_acceleration * Height


def MECHANICAL_ENERGY(Kinetic_Energy: float, Gravitational_Potential_Energy: float):
    """Calculate total mechanical energy.

    Args:
        Kinetic_Energy: Kinetic energy in joules (J).
        Gravitational_Potential_Energy: Gravitational potential energy in joules (J).

    Returns:
        Total mechanical energy in joules (J).
    """
    return Kinetic_Energy + Gravitational_Potential_Energy


def ELASTIC_POTENTIAL_ENERGY(k: float, x: float):
    """Calculate elastic potential energy of a spring.

    Args:
        k: Spring constant in newtons per meter (N/m).
        x: Displacement from equilibrium in meters (m).

    Returns:
        Elastic potential energy in joules (J).
    """
    return 0.5 * k * (x ** 2)
