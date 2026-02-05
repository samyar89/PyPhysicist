import numpy as np

from .mechanics import (
    momentum,
    newton_second_law,
    spring_potential_energy,
    centripetal_acceleration,
    centripetal_force,
)
from .electromagnetism import (
    coulomb_force,
    electric_field,
    capacitance,
    resistance_series,
    resistance_parallel,
)
from .thermodynamics import (
    ideal_gas_pressure,
    heat_capacity,
    entropy_change,
)
from .waves_optics import (
    frequency,
    wavelength,
    wave_power,
    refractive_index,
)
from .relativity import (
    time_dilation,
    length_contraction,
    relativistic_energy,
)


def V(I: float, R: float):
    """Calculate voltage from current and resistance.

    Supports scalar or NumPy array-like inputs.

    Args:
        I: Electric current in amperes (A).
        R: Electrical resistance in ohms (Ω).

    Returns:
        Voltage in volts (V).
    """
    I = np.asarray(I)
    R = np.asarray(R)
    return I * R

def I(V: float, R: float):
    """Calculate current from voltage and resistance.

    Supports scalar or NumPy array-like inputs.

    Args:
        V: Voltage in volts (V).
        R: Electrical resistance in ohms (Ω).

    Returns:
        Electric current in amperes (A).
    """
    V = np.asarray(V)
    R = np.asarray(R)
    return V / R

def R(V: float, I: float):
    """Calculate resistance from voltage and current.

    Supports scalar or NumPy array-like inputs.

    Args:
        V: Voltage in volts (V).
        I: Electric current in amperes (A).

    Returns:
        Electrical resistance in ohms (Ω).
    """
    V = np.asarray(V)
    I = np.asarray(I)
    return V / I

def Velocity(d: float, t: float):
    """Calculate velocity from distance and time.

    Supports scalar or NumPy array-like inputs.

    Args:
        d: Distance in meters (m).
        t: Time in seconds (s).

    Returns:
        Velocity in meters per second (m/s).
    """
    d = np.asarray(d)
    t = np.asarray(t)
    return d / t

def F(m: float, a: float):
    """Calculate force from mass and acceleration.

    Supports scalar or NumPy array-like inputs.

    Args:
        m: Mass in kilograms (kg).
        a: Acceleration in meters per second squared (m/s²).

    Returns:
        Force in newtons (N).
    """
    m = np.asarray(m)
    a = np.asarray(a)
    return m * a
    
def Weight(m: float, g: float):
    """Calculate weight from mass and gravitational acceleration.

    Supports scalar or NumPy array-like inputs.

    Args:
        m: Mass in kilograms (kg).
        g: Gravitational acceleration in meters per second squared (m/s²).

    Returns:
        Weight in newtons (N).
    """
    m = np.asarray(m)
    g = np.asarray(g)
    return m * g

def Work(F: float, d: float):
    """Calculate work from force and displacement.

    Supports scalar or NumPy array-like inputs.

    Args:
        F: Force in newtons (N).
        d: Displacement in meters (m).

    Returns:
        Work in joules (J).
    """
    F = np.asarray(F)
    d = np.asarray(d)
    return F * d

def Schwarzschild_radius(m: float):
    """Calculate the Schwarzschild radius for a given mass.

    Supports scalar or NumPy array-like inputs.

    Args:
        m: Mass in kilograms (kg).

    Returns:
        Schwarzschild radius in meters (m).
    """
    m = np.asarray(m)
    G = 6.67430 * (10 ** -11)
    c = 299792458

    return (2 * G * m) / (c ** 2)
