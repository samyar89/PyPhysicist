"""Validation utilities for classical astrodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity, require_vector_quantity, vector_norm
from .core import CelestialBody, OrbitState
from .elements import state_to_elements
from .invariants import angular_momentum_vector, specific_orbital_energy
from .theorems import vis_viva


@dataclass(frozen=True)
class ValidationReport:
    """Structured report for diagnostic checks."""

    checks: dict


def validate_conservation(body: CelestialBody, states: Iterable[OrbitState]) -> ValidationReport:
    """Check conservation of energy and angular momentum over states."""
    energies = []
    angular_momenta = []
    for state in states:
        energies.append(specific_orbital_energy(body, state).value)
        angular_momenta.append(angular_momentum_vector(state).value)
    energies = np.array(energies)
    angular_momenta = np.array(angular_momenta)

    energy_dev = (energies - energies[0]) / energies[0]
    h_norms = np.linalg.norm(angular_momenta, axis=1)
    h_dev = (h_norms - h_norms[0]) / h_norms[0]

    checks = {
        "specific_energy": {
            "initial": Quantity(float(energies[0]), "m^2/s^2"),
            "max_relative_deviation": float(np.max(np.abs(energy_dev))),
        },
        "angular_momentum": {
            "initial": Quantity(angular_momenta[0], "m^2/s"),
            "max_relative_deviation": float(np.max(np.abs(h_dev))),
        },
    }
    return ValidationReport(checks=checks)


def validate_keplerian_constraints(body: CelestialBody, state: OrbitState) -> ValidationReport:
    """Verify that the state satisfies vis-viva and conic constraints."""
    elements = state_to_elements(body, state)
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = vector_norm(r_vec.value)
    v = vector_norm(v_vec.value)

    v2_expected = vis_viva(
        body,
        Quantity(r, "m"),
        Quantity(elements.semi_major_axis, "m"),
    )["velocity_squared"].value

    residual = v**2 - v2_expected
    checks = {
        "vis_viva_residual": Quantity(residual, "m^2/s^2"),
        "eccentricity": elements.eccentricity,
        "orbit_type": (
            "elliptic"
            if elements.eccentricity < 1
            else "parabolic"
            if elements.eccentricity == 1
            else "hyperbolic"
        ),
    }
    return ValidationReport(checks=checks)


def validate_regime(body: CelestialBody, state: OrbitState, regime: str) -> ValidationReport:
    """Validate that a state remains within an assumed regime."""
    energy = specific_orbital_energy(body, state).value
    r_vec = require_vector_quantity(state.position, "m", name="position")
    r = vector_norm(r_vec.value)
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    escape_speed = np.sqrt(2 * mu.value / r)
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    speed = vector_norm(v_vec.value)

    checks = {
        "regime": regime,
        "specific_energy": Quantity(energy, "m^2/s^2"),
        "speed": Quantity(speed, "m/s"),
        "escape_speed": Quantity(escape_speed, "m/s"),
    }

    if regime == "bound":
        checks["satisfies"] = bool(energy < 0)
    elif regime == "escape":
        checks["satisfies"] = bool(speed >= escape_speed)
    else:
        checks["satisfies"] = None

    return ValidationReport(checks=checks)


__all__ = [
    "ValidationReport",
    "validate_conservation",
    "validate_keplerian_constraints",
    "validate_regime",
]
