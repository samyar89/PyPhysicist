"""Circuit relationships and equivalent components."""

import numpy as np


def voltage(current: float, resistance: float):
    """Calculate voltage from current and resistance (Ohm's law)."""
    current = np.asarray(current)
    resistance = np.asarray(resistance)
    return current * resistance


def current(voltage_value: float, resistance: float):
    """Calculate current from voltage and resistance."""
    voltage_value = np.asarray(voltage_value)
    resistance = np.asarray(resistance)
    return voltage_value / resistance


def resistance(voltage_value: float, current_value: float):
    """Calculate resistance from voltage and current."""
    voltage_value = np.asarray(voltage_value)
    current_value = np.asarray(current_value)
    return voltage_value / current_value


def resistance_series(*resistances: float):
    """Calculate equivalent resistance for series resistors."""
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for value in resistances:
        total = total + np.asarray(value)
    return total


def resistance_parallel(*resistances: float):
    """Calculate equivalent resistance for parallel resistors."""
    if not resistances:
        return np.asarray(0.0)
    total = np.asarray(0.0)
    for value in resistances:
        total = total + (1 / np.asarray(value))
    return 1 / total


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
