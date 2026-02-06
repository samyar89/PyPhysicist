"""Conserved quantities in classical two-body motion."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity, require_vector_quantity, vector_norm
from .core import CelestialBody, OrbitState


def specific_orbital_energy(body: CelestialBody, state: OrbitState) -> Quantity:
    """Return specific orbital energy ε = v^2/2 - μ/r."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = vector_norm(r_vec.value)
    v = vector_norm(v_vec.value)
    energy = 0.5 * v**2 - mu.value / r
    return Quantity(energy, "m^2/s^2")


def angular_momentum_vector(state: OrbitState) -> Quantity:
    """Return specific angular momentum vector h = r × v."""
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    h_vec = np.cross(r_vec.value, v_vec.value)
    return Quantity(h_vec, "m^2/s")


def laplace_runge_lenz_vector(body: CelestialBody, state: OrbitState) -> Quantity:
    """Return Laplace–Runge–Lenz vector e = (v × h)/μ - r/|r|."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    h_vec = np.cross(r_vec.value, v_vec.value)
    e_vec = np.cross(v_vec.value, h_vec) / mu.value - r_vec.value / vector_norm(
        r_vec.value
    )
    return Quantity(e_vec, "1")


def areal_velocity(state: OrbitState) -> Quantity:
    """Return areal velocity |r × v| / 2."""
    h = angular_momentum_vector(state).value
    return Quantity(0.5 * np.linalg.norm(h), "m^2/s")


__all__ = [
    "specific_orbital_energy",
    "angular_momentum_vector",
    "laplace_runge_lenz_vector",
    "areal_velocity",
]
