"""Dimensionless analysis utilities for orbital dynamics."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity


def energy_ratio(mu: Quantity, radius: Quantity, speed: Quantity) -> dict:
    """Return dimensionless specific energy ratio 2εr/μ."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    v_q = require_scalar_quantity(speed, "m/s", name="speed")
    energy = 0.5 * v_q.value**2 - mu_q.value / r_q.value
    ratio = 2 * energy * r_q.value / mu_q.value
    return {
        "assumptions": "Two-body Newtonian gravity.",
        "energy_ratio": Quantity(ratio, "1"),
        "expression": "2εr/μ",
    }


def angular_momentum_ratio(mu: Quantity, radius: Quantity, speed: Quantity) -> dict:
    """Return dimensionless angular momentum ratio h / sqrt(μ r)."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    v_q = require_scalar_quantity(speed, "m/s", name="speed")
    h = r_q.value * v_q.value
    ratio = h / np.sqrt(mu_q.value * r_q.value)
    return {
        "assumptions": "Planar circular reference scaling.",
        "angular_momentum_ratio": Quantity(ratio, "1"),
        "expression": "h / sqrt(μ r)",
    }


def escape_parameter(mu: Quantity, radius: Quantity, speed: Quantity) -> dict:
    """Return escape parameter v^2 / v_esc^2."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    v_q = require_scalar_quantity(speed, "m/s", name="speed")
    v_escape = np.sqrt(2 * mu_q.value / r_q.value)
    ratio = (v_q.value / v_escape) ** 2
    return {
        "assumptions": "Two-body Newtonian escape condition.",
        "escape_parameter": Quantity(ratio, "1"),
        "expression": "v^2 / v_esc^2",
    }


def normalized_time_scale(mu: Quantity, radius: Quantity, time: Quantity) -> dict:
    """Return nondimensionalized time τ = t / sqrt(r^3/μ)."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    t_q = require_scalar_quantity(time, "s", name="time")
    scale = np.sqrt(r_q.value**3 / mu_q.value)
    tau = t_q.value / scale
    return {
        "assumptions": "Two-body Newtonian scaling.",
        "normalized_time": Quantity(tau, "1"),
        "expression": "τ = t / sqrt(r^3/μ)",
    }


__all__ = [
    "energy_ratio",
    "angular_momentum_ratio",
    "escape_parameter",
    "normalized_time_scale",
]
