"""Wave relationships."""

import numpy as np


def frequency(wave_speed: float, wavelength_value: float):
    """Calculate wave frequency."""
    wave_speed = np.asarray(wave_speed)
    wavelength_value = np.asarray(wavelength_value)
    return wave_speed / wavelength_value


def wavelength(wave_speed: float, frequency_hz: float):
    """Calculate wavelength from wave speed and frequency."""
    wave_speed = np.asarray(wave_speed)
    frequency_hz = np.asarray(frequency_hz)
    return wave_speed / frequency_hz


def wave_power(energy: float, time: float):
    """Calculate wave power."""
    energy = np.asarray(energy)
    time = np.asarray(time)
    return energy / time


__all__ = ["frequency", "wavelength", "wave_power"]
