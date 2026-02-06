"""Kepler's laws and classical orbital theorems."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity, require_vector_quantity, vector_norm
from .core import CelestialBody, OrbitState
from .invariants import areal_velocity, specific_orbital_energy


def kepler_first_law(eccentricity: float) -> dict:
    """Return Kepler's first law statement for conic sections."""
    orbit_type = (
        "elliptic" if eccentricity < 1 else "parabolic" if eccentricity == 1 else "hyperbolic"
    )
    return {
        "assumptions": "Two-body Newtonian gravity with point mass center.",
        "statement": "Orbits are conic sections with the central body at one focus.",
        "orbit_type": orbit_type,
    }


def kepler_second_law(state: OrbitState) -> dict:
    """Return areal velocity from Kepler's second law."""
    area_rate = areal_velocity(state)
    return {
        "assumptions": "Central force, planar motion.",
        "areal_velocity": area_rate,
        "statement": "Equal areas are swept in equal times.",
    }


def kepler_third_law(body: CelestialBody, semi_major_axis: Quantity) -> dict:
    """Return period and mean motion from Kepler's third law."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    a = require_scalar_quantity(semi_major_axis, "m", name="semi_major_axis")
    n = np.sqrt(mu.value / a.value**3)
    period = 2 * np.pi / n
    return {
        "assumptions": "Bound Keplerian orbit, two-body Newtonian gravity.",
        "mean_motion": Quantity(n, "1/s"),
        "period": Quantity(period, "s"),
    }


def virial_theorem(body: CelestialBody, state: OrbitState) -> dict:
    """Return orbital virial theorem diagnostic for bound motion."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = vector_norm(r_vec.value)
    v = vector_norm(v_vec.value)
    kinetic = 0.5 * v**2
    potential = -mu.value / r
    return {
        "assumptions": "Bound orbit time-averaged over one period.",
        "kinetic_specific": Quantity(kinetic, "m^2/s^2"),
        "potential_specific": Quantity(potential, "m^2/s^2"),
        "statement": "2⟨T⟩ + ⟨U⟩ = 0 for inverse-square law.",
    }


def escape_velocity(body: CelestialBody, radius: Quantity) -> dict:
    """Return escape velocity at a given radius."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r = require_scalar_quantity(radius, "m", name="radius")
    v = np.sqrt(2 * mu.value / r.value)
    return {
        "assumptions": "Two-body Newtonian gravity, zero energy boundary.",
        "escape_velocity": Quantity(v, "m/s"),
        "statement": "v >= sqrt(2 μ / r) for escape.",
    }


def vis_viva(body: CelestialBody, radius: Quantity, semi_major_axis: Quantity) -> dict:
    """Return vis-viva equation for orbital speed."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r = require_scalar_quantity(radius, "m", name="radius")
    a = require_scalar_quantity(semi_major_axis, "m", name="semi_major_axis")
    v2 = mu.value * (2 / r.value - 1 / a.value)
    return {
        "assumptions": "Two-body Newtonian gravity.",
        "velocity_squared": Quantity(v2, "m^2/s^2"),
        "statement": "v^2 = μ(2/r - 1/a).",
    }


def energy_condition(state: OrbitState, body: CelestialBody) -> dict:
    """Classify orbit type by specific orbital energy."""
    energy = specific_orbital_energy(body, state)
    classification = (
        "bound" if energy.value < 0 else "parabolic" if energy.value == 0 else "unbound"
    )
    return {
        "specific_energy": energy,
        "classification": classification,
    }


__all__ = [
    "kepler_first_law",
    "kepler_second_law",
    "kepler_third_law",
    "virial_theorem",
    "escape_velocity",
    "vis_viva",
    "energy_condition",
]
