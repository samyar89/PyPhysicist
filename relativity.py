"""Relativity formulas."""


def time_dilation(proper_time: float, velocity: float, c: float):
    """
    proper_time: s
    velocity: m/s
    c: m/s

    In this case, the dilated time is obtained in seconds.
    """
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_time * gamma


def length_contraction(proper_length: float, velocity: float, c: float):
    """
    proper_length: m
    velocity: m/s
    c: m/s

    In this case, the contracted length is obtained in meters.
    """
    gamma = 1 / ((1 - (velocity ** 2) / (c ** 2)) ** 0.5)
    return proper_length / gamma


def relativistic_energy(mass: float, c: float):
    """
    mass: kg
    c: m/s

    In this case, the energy is obtained in Joules.
    """
    return mass * (c ** 2)
