"""Wave and optics formulas."""


def frequency(wave_speed: float, wavelength: float):
    """
    wave_speed: m/s
    wavelength: m

    In this case, the frequency is obtained in Hz.
    """
    return wave_speed / wavelength


def wavelength(wave_speed: float, frequency_hz: float):
    """
    wave_speed: m/s
    frequency_hz: Hz

    In this case, the wavelength is obtained in meters.
    """
    return wave_speed / frequency_hz


def wave_power(energy: float, time: float):
    """
    energy: J
    time: s

    In this case, the wave power is obtained in Watts.
    """
    return energy / time


def refractive_index(c: float, v: float):
    """
    c: m/s
    v: m/s

    In this case, the refractive index is unitless.
    """
    return c / v
