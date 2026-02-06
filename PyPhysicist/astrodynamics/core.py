"""Core abstractions for classical astrodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import (
    require_scalar_quantity,
    require_vector_quantity,
    unit_dimensions,
    vector_norm,
)
from . import invariants


@dataclass(frozen=True)
class CelestialBody:
    """Idealized central body for two-body motion.

    Args:
        gravitational_parameter: Standard gravitational parameter μ = GM.
        mass: Body mass.
        radius: Mean radius of the body.
        reference_frame: Name of the inertial reference frame.
        shape_assumption: "point_mass" or "spherical".
    """

    gravitational_parameter: Quantity
    mass: Quantity
    radius: Quantity
    reference_frame: str = "inertial"
    shape_assumption: str = "spherical"

    def __post_init__(self) -> None:
        require_scalar_quantity(
            self.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
        )
        require_scalar_quantity(self.mass, "kg", name="mass")
        require_scalar_quantity(self.radius, "m", name="radius")
        if self.shape_assumption not in {"point_mass", "spherical"}:
            raise ValueError("shape_assumption must be 'point_mass' or 'spherical'.")


@dataclass(frozen=True)
class OrbitState:
    """State vector for orbital motion."""

    position: Quantity
    velocity: Quantity
    time: Quantity
    reference_frame: str = "inertial"

    def __post_init__(self) -> None:
        require_vector_quantity(self.position, "m", name="position")
        require_vector_quantity(self.velocity, "m/s", name="velocity")
        require_scalar_quantity(self.time, "s", name="time")

    @property
    def r_mag(self) -> Quantity:
        magnitude = vector_norm(self.position.value)
        return Quantity(magnitude, "m")

    @property
    def v_mag(self) -> Quantity:
        magnitude = vector_norm(self.velocity.value)
        return Quantity(magnitude, "m/s")


@dataclass(frozen=True)
class Orbit:
    """Orbit object for a test particle around a central body."""

    central_body: CelestialBody
    state: OrbitState
    elements: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.state.reference_frame != self.central_body.reference_frame:
            raise ValueError(
                "OrbitState reference frame must match the CelestialBody reference frame."
            )

    @property
    def specific_energy(self) -> Quantity:
        return invariants.specific_orbital_energy(self.central_body, self.state)

    @property
    def angular_momentum(self) -> Quantity:
        return invariants.angular_momentum_vector(self.state)

    @property
    def orbital_plane_normal(self) -> np.ndarray:
        h = self.angular_momentum.value
        return h / np.linalg.norm(h)

    @property
    def conserved_quantities(self) -> dict:
        return {
            "specific_energy": self.specific_energy,
            "angular_momentum": self.angular_momentum,
        }

    @property
    def assumptions(self) -> dict:
        return {
            "reference_frame": self.central_body.reference_frame,
            "shape_assumption": self.central_body.shape_assumption,
            "gravity_model": "Newtonian point-mass",
            "dimensional_basis": unit_dimensions(self.central_body.gravitational_parameter.unit),
        }


__all__ = ["CelestialBody", "OrbitState", "Orbit"]
