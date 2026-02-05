"""Mechanical energy formulas."""

import numpy as np


def kinetic_energy(mass: float, velocity: float):
    """Calculate kinetic energy."""
    mass = np.asarray(mass)
    velocity = np.asarray(velocity)
    return 0.5 * mass * (velocity ** 2)


def gravitational_potential_energy(mass: float, gravity: float, height: float):
    """Calculate gravitational potential energy."""
    mass = np.asarray(mass)
    gravity = np.asarray(gravity)
    height = np.asarray(height)
    return mass * gravity * height


def mechanical_energy(kinetic: float, potential: float):
    """Calculate total mechanical energy."""
    kinetic = np.asarray(kinetic)
    potential = np.asarray(potential)
    return kinetic + potential


def spring_potential_energy(spring_constant: float, displacement: float):
    """Calculate potential energy stored in a spring."""
    spring_constant = np.asarray(spring_constant)
    displacement = np.asarray(displacement)
    return 0.5 * spring_constant * (displacement ** 2)


def work(force_value: float, displacement: float):
    """Calculate work from force and displacement."""
    force_value = np.asarray(force_value)
    displacement = np.asarray(displacement)
    return force_value * displacement


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
