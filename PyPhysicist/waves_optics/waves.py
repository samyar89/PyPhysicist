"""Wave relationships."""

from ..units import coerce_value, wrap_quantity


def frequency(wave_speed: float, wavelength_value: float):
    """Calculate wave frequency."""
    wave_speed_value, _ = coerce_value(wave_speed, "m/s", name="wave_speed")
    wavelength_value_value, _ = coerce_value(wavelength_value, "m", name="wavelength")
    result = wave_speed_value / wavelength_value_value
    return wrap_quantity(result, "Hz", wave_speed, wavelength_value)


def wavelength(wave_speed: float, frequency_hz: float):
    """Calculate wavelength from wave speed and frequency."""
    wave_speed_value, _ = coerce_value(wave_speed, "m/s", name="wave_speed")
    frequency_value, _ = coerce_value(frequency_hz, "Hz", name="frequency")
    result = wave_speed_value / frequency_value
    return wrap_quantity(result, "m", wave_speed, frequency_hz)


def wave_power(energy: float, time: float):
    """Calculate wave power."""
    energy_value, _ = coerce_value(energy, "J", name="energy")
    time_value, _ = coerce_value(time, "s", name="time")
    result = energy_value / time_value
    return wrap_quantity(result, "W", energy, time)


__all__ = ["frequency", "wavelength", "wave_power"]
