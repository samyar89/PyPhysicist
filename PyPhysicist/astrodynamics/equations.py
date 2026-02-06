"""Governing equations for Newtonian orbital motion."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity, require_vector_quantity, vector_norm
from .core import CelestialBody, OrbitState


def two_body_equation(
    body: CelestialBody,
    state: OrbitState,
    acceleration: Quantity | None = None,
) -> dict:
    """Return the two-body equation residual and expected acceleration."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    r = vector_norm(r_vec.value)
    expected = -mu.value * r_vec.value / r**3
    expected_quantity = Quantity(expected, "m/s^2")
    residual = None
    if acceleration is not None:
        acc_vec = require_vector_quantity(acceleration, "m/s^2", name="acceleration")
        residual = Quantity(acc_vec.value - expected, "m/s^2")
    return {
        "equation": "r_ddot + μ r / |r|^3 = 0",
        "expected_acceleration": expected_quantity,
        "residual": residual,
    }


def reduced_mass_equation(
    body: CelestialBody,
    state: OrbitState,
    secondary_mass: Quantity | None = None,
) -> dict:
    """Reduced-mass formulation for relative motion."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    if secondary_mass is not None:
        m2 = require_scalar_quantity(secondary_mass, "kg", name="secondary_mass")
        m1 = require_scalar_quantity(body.mass, "kg", name="mass")
        mu_total = Quantity(mu.value * (m1.value + m2.value) / m1.value, "m^3/s^2")
    else:
        mu_total = mu
    r_vec = require_vector_quantity(state.position, "m", name="position")
    r = vector_norm(r_vec.value)
    expected = -mu_total.value * r_vec.value / r**3
    return {
        "equation": "r_ddot = -G (m1+m2) r / |r|^3",
        "effective_gravitational_parameter": mu_total,
        "expected_acceleration": Quantity(expected, "m/s^2"),
    }


def central_force_equation(body: CelestialBody, state: OrbitState) -> dict:
    """Central-force equation in vector and polar form."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = vector_norm(r_vec.value)
    h_vec = np.cross(r_vec.value, v_vec.value)
    h = vector_norm(h_vec)
    radial_equation = "r_ddot - h^2 / r^3 = -μ / r^2"
    angular_equation = "d/dt (r^2 θ_dot) = 0"
    return {
        "vector_equation": "r_ddot = -μ r / |r|^3",
        "radial_equation": radial_equation,
        "angular_equation": angular_equation,
        "specific_angular_momentum": Quantity(h_vec, "m^2/s"),
        "orbit_parameter": Quantity(h**2 / mu.value, "m"),
    }


def polar_component_equations(body: CelestialBody, state: OrbitState) -> dict:
    """Return radial and angular equations in polar coordinates."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = vector_norm(r_vec.value)
    h_vec = np.cross(r_vec.value, v_vec.value)
    h = vector_norm(h_vec)
    return {
        "radial_equation": "r_ddot = r θ_dot^2 - μ / r^2",
        "angular_equation": "r^2 θ_dot = h",
        "specific_angular_momentum": Quantity(h_vec, "m^2/s"),
        "gravity_term": Quantity(mu.value / r**2, "m/s^2"),
    }


__all__ = [
    "two_body_equation",
    "reduced_mass_equation",
    "central_force_equation",
    "polar_component_equations",
]
