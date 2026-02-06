"""Mechanical energy formulas."""

from ..units import coerce_value, ensure_same_units, wrap_quantity


def kinetic_energy(mass: float, velocity: float):
    """Calculate kinetic energy."""
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    velocity_value, _ = coerce_value(velocity, "m/s", name="velocity")
    result = 0.5 * mass_value * (velocity_value ** 2)
    return wrap_quantity(result, "J", mass, velocity)


def gravitational_potential_energy(mass: float, gravity: float, height: float):
    """Calculate gravitational potential energy."""
    mass_value, _ = coerce_value(mass, "kg", name="mass")
    gravity_value, _ = coerce_value(gravity, "m/s^2", name="gravity")
    height_value, _ = coerce_value(height, "m", name="height")
    result = mass_value * gravity_value * height_value
    return wrap_quantity(result, "J", mass, gravity, height)


def mechanical_energy(kinetic: float, potential: float):
    """Calculate total mechanical energy."""
    kinetic_value, kinetic_unit = coerce_value(kinetic, "J", name="kinetic")
    potential_value, potential_unit = coerce_value(potential, "J", name="potential")
    if kinetic_unit and potential_unit:
        ensure_same_units(kinetic_unit, potential_unit, name_a="kinetic", name_b="potential")
    result = kinetic_value + potential_value
    return wrap_quantity(result, "J", kinetic, potential)


def spring_potential_energy(spring_constant: float, displacement: float):
    """Calculate potential energy stored in a spring."""
    spring_value, _ = coerce_value(spring_constant, "N/m", name="spring_constant")
    displacement_value, _ = coerce_value(displacement, "m", name="displacement")
    result = 0.5 * spring_value * (displacement_value ** 2)
    return wrap_quantity(result, "J", spring_constant, displacement)


def work(force_value: float, displacement: float):
    """Calculate work from force and displacement."""
    force_value_value, _ = coerce_value(force_value, "N", name="force")
    displacement_value, _ = coerce_value(displacement, "m", name="displacement")
    result = force_value_value * displacement_value
    return wrap_quantity(result, "J", force_value, displacement)


elastic_potential_energy = spring_potential_energy

KINETIC_ENERGY = kinetic_energy
GRAVITATIONAL_POTENTIAL_ENERGY = gravitational_potential_energy
MECHANICAL_ENERGY = mechanical_energy
ELASTIC_POTENTIAL_ENERGY = spring_potential_energy
Work = work

__all__ = [
    "kinetic_energy",
    "gravitational_potential_energy",
    "mechanical_energy",
    "spring_potential_energy",
    "elastic_potential_energy",
    "work",
    "KINETIC_ENERGY",
    "GRAVITATIONAL_POTENTIAL_ENERGY",
    "MECHANICAL_ENERGY",
    "ELASTIC_POTENTIAL_ENERGY",
    "Work",
]
