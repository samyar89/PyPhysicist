"""Relativity formulas."""


def time_dilation(proper_time: float, velocity: float, c: float):
    """Calculate time dilation.

    Args:
        proper_time: Proper time in seconds (s).
        velocity: Relative velocity in meters per second (m/s).
        c: Speed of light in meters per second (m/s).

    Returns:
        Dilated time in seconds (s).
    """
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_time * gamma


def length_contraction(proper_length: float, velocity: float, c: float):
    """Calculate length contraction.

    Args:
        proper_length: Proper length in meters (m).
        velocity: Relative velocity in meters per second (m/s).
        c: Speed of light in meters per second (m/s).

    Returns:
        Contracted length in meters (m).
    """
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_length / gamma


def relativistic_energy(mass: float, c: float):
    """Calculate rest energy.

    Args:
        mass: Mass in kilograms (kg).
        c: Speed of light in meters per second (m/s).

    Returns:
        Rest energy in joules (J).
    """
    return mass * (c ** 2)
