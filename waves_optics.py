"""Wave and optics formulas."""


def frequency(wave_speed: float, wavelength: float):
    """Calculate wave frequency.

    Args:
        wave_speed: Wave speed in meters per second (m/s).
        wavelength: Wavelength in meters (m).

    Returns:
        Frequency in hertz (Hz).
    """
    return wave_speed / wavelength


def wavelength(wave_speed: float, frequency_hz: float):
    """Calculate wavelength from wave speed and frequency.

    Args:
        wave_speed: Wave speed in meters per second (m/s).
        frequency_hz: Frequency in hertz (Hz).

    Returns:
        Wavelength in meters (m).
    """
    return wave_speed / frequency_hz


def wave_power(energy: float, time: float):
    """Calculate wave power.

    Args:
        energy: Energy in joules (J).
        time: Time interval in seconds (s).

    Returns:
        Power in watts (W).
    """
    return energy / time


def refractive_index(c: float, v: float):
    """Calculate refractive index.

    Args:
        c: Speed of light in vacuum in meters per second (m/s).
        v: Speed of light in the medium in meters per second (m/s).

    Returns:
        Refractive index (unitless).
    """
    return c / v
