"""Circuit relationships and equivalent components."""

import numpy as np

from ..units import coerce_value, wrap_quantity


def voltage(current: float, resistance: float):
    """Calculate voltage from current and resistance (Ohm's law)."""
    current_value, _ = coerce_value(current, "A", name="current")
    resistance_value, _ = coerce_value(resistance, "ohm", name="resistance")
    result = current_value * resistance_value
    return wrap_quantity(result, "V", current, resistance)


def current(voltage_value: float, resistance: float):
    """Calculate current from voltage and resistance."""
    voltage_value_value, _ = coerce_value(voltage_value, "V", name="voltage")
    resistance_value, _ = coerce_value(resistance, "ohm", name="resistance")
    result = voltage_value_value / resistance_value
    return wrap_quantity(result, "A", voltage_value, resistance)


def resistance(voltage_value: float, current_value: float):
    """Calculate resistance from voltage and current."""
    voltage_value_value, _ = coerce_value(voltage_value, "V", name="voltage")
    current_value_value, _ = coerce_value(current_value, "A", name="current")
    result = voltage_value_value / current_value_value
    return wrap_quantity(result, "ohm", voltage_value, current_value)


def resistance_series(*resistances: float):
    """Calculate equivalent resistance for series resistors."""
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for value in resistances:
        resistance_value, _ = coerce_value(value, "ohm", name="resistance")
        total = total + resistance_value
    return wrap_quantity(total, "ohm", *resistances)


def resistance_parallel(*resistances: float):
    """Calculate equivalent resistance for parallel resistors."""
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for value in resistances:
        resistance_value, _ = coerce_value(value, "ohm", name="resistance")
        total = total + (1 / resistance_value)
    result = 1 / total
    return wrap_quantity(result, "ohm", *resistances)


V = voltage
I = current
R = resistance

__all__ = [
    "voltage",
    "current",
    "resistance",
    "resistance_series",
    "resistance_parallel",
    "V",
    "I",
    "R",
]
