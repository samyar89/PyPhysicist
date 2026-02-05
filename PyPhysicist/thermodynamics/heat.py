"""Heat and entropy relationships."""

import numpy as np


def heat_capacity(heat: float, mass: float, delta_t: float):
    """Calculate specific heat capacity."""
    heat = np.asarray(heat)
    mass = np.asarray(mass)
    delta_t = np.asarray(delta_t)
    return heat / (mass * delta_t)


def entropy_change(heat: float, temperature: float):
    """Calculate entropy change."""
    heat = np.asarray(heat)
    temperature = np.asarray(temperature)
    return heat / temperature


__all__ = ["heat_capacity", "entropy_change"]
