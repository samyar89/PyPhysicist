"""Wave and optics formulas."""

import numpy as np


def frequency(wave_speed: float, wavelength: float):
    """Calculate wave frequency.

    Supports scalar or NumPy array-like inputs.

    Args:
        wave_speed: Wave speed in meters per second (m/s).
        wavelength: Wavelength in meters (m).

    Returns:
        Frequency in hertz (Hz).
    """
    wave_speed = np.asarray(wave_speed)
    wavelength = np.asarray(wavelength)
    return wave_speed / wavelength


def wavelength(wave_speed: float, frequency_hz: float):
    """Calculate wavelength from wave speed and frequency.

    Supports scalar or NumPy array-like inputs.

    Args:
        wave_speed: Wave speed in meters per second (m/s).
        frequency_hz: Frequency in hertz (Hz).

    Returns:
        Wavelength in meters (m).
    """
    wave_speed = np.asarray(wave_speed)
    frequency_hz = np.asarray(frequency_hz)
    return wave_speed / frequency_hz


def wave_power(energy: float, time: float):
    """Calculate wave power.

    Supports scalar or NumPy array-like inputs.

    Args:
        energy: Energy in joules (J).
        time: Time interval in seconds (s).

    Returns:
        Power in watts (W).
    """
    energy = np.asarray(energy)
    time = np.asarray(time)
    return energy / time


def refractive_index(c: float, v: float):
    """Calculate refractive index.

    Supports scalar or NumPy array-like inputs.

    Args:
        c: Speed of light in vacuum in meters per second (m/s).
        v: Speed of light in the medium in meters per second (m/s).

    Returns:
        Refractive index (unitless).
    """
    c = np.asarray(c)
    v = np.asarray(v)
    return c / v
