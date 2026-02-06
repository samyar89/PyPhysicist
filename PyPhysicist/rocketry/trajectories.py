"""Analytical and idealized rocket trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


@dataclass(frozen=True)
class TrajectorySolution:
    """Container for analytical trajectory results."""

    description: str
    formula: str
    value: Any
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def vertical_ascent_velocity(time: Quantity | float, *, acceleration: Quantity | float, initial_velocity: Quantity | float = 0.0) -> TrajectorySolution:
    """Constant-acceleration vertical ascent velocity."""

    a_value, _ = coerce_value(acceleration, "m/s^2", name="acceleration")
    t_value, _ = coerce_value(time, "s", name="time")
    v0_value, _ = coerce_value(initial_velocity, "m/s", name="initial_velocity")
    velocity = v0_value + a_value * t_value
    result = wrap_quantity(velocity, "m/s", acceleration, time, initial_velocity)
    return TrajectorySolution(
        "vertical ascent velocity",
        "v = v0 + a t",
        result,
        ("constant acceleration",),
    )


def constant_thrust_planar_range(time: Quantity | float, *, acceleration: Quantity | float, flight_path_angle: Quantity | float) -> TrajectorySolution:
    """Planar range under constant acceleration and fixed flight-path angle."""

    a_value, _ = coerce_value(acceleration, "m/s^2", name="acceleration")
    t_value, _ = coerce_value(time, "s", name="time")
    gamma_value, _ = coerce_value(flight_path_angle, "rad", name="flight_path_angle")
    range_value = 0.5 * a_value * np.cos(gamma_value) * t_value**2
    result = wrap_quantity(range_value, "m", acceleration, time, flight_path_angle)
    return TrajectorySolution(
        "constant-thrust planar range",
        "x = 1/2 a cos(γ) t^2",
        result,
        ("constant thrust", "fixed flight-path angle"),
    )


def vacuum_ascent_altitude(time: Quantity | float, *, acceleration: Quantity | float, initial_velocity: Quantity | float = 0.0) -> TrajectorySolution:
    """Vacuum ascent altitude under constant acceleration."""

    a_value, _ = coerce_value(acceleration, "m/s^2", name="acceleration")
    t_value, _ = coerce_value(time, "s", name="time")
    v0_value, _ = coerce_value(initial_velocity, "m/s", name="initial_velocity")
    altitude = v0_value * t_value + 0.5 * a_value * t_value**2
    result = wrap_quantity(altitude, "m", acceleration, time, initial_velocity)
    return TrajectorySolution(
        "vacuum ascent altitude",
        "h = v0 t + 1/2 a t^2",
        result,
        ("vacuum", "constant acceleration"),
    )


def orbital_insertion_condition(mu: Quantity | float, radius: Quantity | float, velocity: Quantity | float) -> TrajectorySolution:
    """Energy and angular momentum condition for circular orbit insertion."""

    mu_value, _ = coerce_value(mu, "m^3/s^2", name="mu")
    r_value, _ = coerce_value(radius, "m", name="radius")
    v_value, _ = coerce_value(velocity, "m/s", name="velocity")
    circular_speed = np.sqrt(mu_value / r_value)
    residual = v_value - circular_speed
    result = wrap_quantity(residual, "m/s", mu, radius, velocity)
    return TrajectorySolution(
        "orbital insertion condition",
        "v - sqrt(μ/r) = 0",
        result,
        ("circular orbit",),
    )


__all__ = [
    "TrajectorySolution",
    "vertical_ascent_velocity",
    "constant_thrust_planar_range",
    "vacuum_ascent_altitude",
    "orbital_insertion_condition",
]
