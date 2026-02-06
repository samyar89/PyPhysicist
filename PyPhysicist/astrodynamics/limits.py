"""Perturbation-free analytical limits and approximations."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity


def small_eccentricity_radius(
    semi_major_axis: Quantity, eccentricity: float, true_anomaly: float
) -> dict:
    """Approximate radius for small eccentricity."""
    a = require_scalar_quantity(semi_major_axis, "m", name="semi_major_axis")
    r = a.value * (1 - eccentricity * np.cos(true_anomaly))
    return {
        "assumptions": "e << 1, first-order expansion.",
        "radius": Quantity(r, "m"),
        "expression": "r ≈ a(1 - e cos θ)",
    }


def nearly_circular_velocity(mu: Quantity, radius: Quantity, eccentricity: float) -> dict:
    """Approximate tangential speed for nearly circular orbits."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    v = np.sqrt(mu_q.value / r_q.value) * (1 + 0.5 * eccentricity)
    return {
        "assumptions": "e << 1, tangential velocity expansion.",
        "velocity": Quantity(v, "m/s"),
        "expression": "v ≈ sqrt(μ/r)(1 + e/2)",
    }


def bound_unbound_comparison(mu: Quantity, radius: Quantity, speed: Quantity) -> dict:
    """Compare bound vs unbound motion by specific energy."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    v_q = require_scalar_quantity(speed, "m/s", name="speed")
    energy = 0.5 * v_q.value**2 - mu_q.value / r_q.value
    classification = (
        "bound" if energy < 0 else "parabolic" if energy == 0 else "unbound"
    )
    return {
        "assumptions": "Two-body Newtonian gravity.",
        "specific_energy": Quantity(energy, "m^2/s^2"),
        "classification": classification,
    }


def scaling_arguments(mu: Quantity, radius: Quantity) -> dict:
    """Return characteristic time and velocity scales."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    time_scale = np.sqrt(r_q.value**3 / mu_q.value)
    velocity_scale = np.sqrt(mu_q.value / r_q.value)
    return {
        "assumptions": "Two-body Newtonian scaling.",
        "time_scale": Quantity(time_scale, "s"),
        "velocity_scale": Quantity(velocity_scale, "m/s"),
        "expression": "t~sqrt(r^3/μ), v~sqrt(μ/r)",
    }


__all__ = [
    "small_eccentricity_radius",
    "nearly_circular_velocity",
    "bound_unbound_comparison",
    "scaling_arguments",
]
