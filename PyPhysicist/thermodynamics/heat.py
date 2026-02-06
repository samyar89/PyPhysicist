"""Heat and entropy relationships."""

from ..units import coerce_value, wrap_quantity


def heat_capacity(heat: float, mass: float, delta_t: float):
    """Calculate specific heat capacity."""
    heat_value, _ = coerce_value(heat, "J", name="heat")
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    delta_t_value, _ = coerce_value(delta_t, "K", name="delta_t", ignore_offset=True)
    result = heat_value / (mass_value * delta_t_value)
    return wrap_quantity(result, "J/kg*K", heat, mass, delta_t)


def entropy_change(heat: float, temperature: float):
    """Calculate entropy change."""
    heat_value, _ = coerce_value(heat, "J", name="heat")
    temperature_value, _ = coerce_value(temperature, "K", name="temperature")
    result = heat_value / temperature_value
    return wrap_quantity(result, "J/K", heat, temperature)


__all__ = ["heat_capacity", "entropy_change"]
