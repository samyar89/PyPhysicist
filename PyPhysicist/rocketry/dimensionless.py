"""Dimensionless groups for rocket analysis."""

from __future__ import annotations

from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


def mass_ratio(initial_mass: Quantity | float, final_mass: Quantity | float) -> Quantity | float:
    m0_value, _ = coerce_value(initial_mass, "kg", name="initial_mass")
    mf_value, _ = coerce_value(final_mass, "kg", name="final_mass")
    ratio = m0_value / mf_value
    return wrap_quantity(ratio, "1", initial_mass, final_mass)


def thrust_to_weight(thrust: Quantity | float, mass: Quantity | float, gravity: Quantity | float) -> Quantity | float:
    thrust_value, _ = coerce_value(thrust, "N", name="thrust")
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    g_value, _ = coerce_value(gravity, "m/s^2", name="gravity")
    ratio = thrust_value / (mass_value * g_value)
    return wrap_quantity(ratio, "1", thrust, mass, gravity)


def characteristic_velocity_ratio(characteristic_velocity: Quantity | float, reference_velocity: Quantity | float) -> Quantity | float:
    v_char, _ = coerce_value(characteristic_velocity, "m/s", name="characteristic_velocity")
    v_ref, _ = coerce_value(reference_velocity, "m/s", name="reference_velocity")
    ratio = v_char / v_ref
    return wrap_quantity(ratio, "1", characteristic_velocity, reference_velocity)


def ballistic_coefficient(mass: Quantity | float, drag_coefficient: float, area: Quantity | float) -> Quantity | float:
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    area_value, _ = coerce_value(area, "m^2", name="area")
    coefficient = mass_value / (drag_coefficient * area_value)
    return wrap_quantity(coefficient, "kg/m^2", mass, area)


__all__ = [
    "mass_ratio",
    "thrust_to_weight",
    "characteristic_velocity_ratio",
    "ballistic_coefficient",
]
