"""Classical orbital elements and geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity, require_vector_quantity, vector_norm
from .core import CelestialBody, OrbitState


@dataclass(frozen=True)
class OrbitalElements:
    """Classical orbital elements."""

    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    argument_of_periapsis: float
    true_anomaly: float
    eccentric_anomaly: float | None
    mean_anomaly: float
    semi_latus_rectum: float
    singularities: tuple[str, ...] = ()


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def _normalize_angle(angle: float) -> float:
    return float(np.mod(angle, 2 * np.pi))


def state_to_elements(body: CelestialBody, state: OrbitState, tol: float = 1e-8) -> OrbitalElements:
    """Convert a state vector to classical orbital elements."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    r_vec = require_vector_quantity(state.position, "m", name="position")
    v_vec = require_vector_quantity(state.velocity, "m/s", name="velocity")
    r = r_vec.value
    v = v_vec.value

    r_norm = vector_norm(r)
    v_norm = vector_norm(v)

    h_vec = np.cross(r, v)
    h_norm = vector_norm(h_vec)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_norm = vector_norm(n_vec)

    e_vec = np.cross(v, h_vec) / mu.value - r / r_norm
    e = vector_norm(e_vec)

    energy = 0.5 * v_norm**2 - mu.value / r_norm
    if abs(e - 1.0) < tol:
        a = np.inf
    else:
        a = -mu.value / (2.0 * energy)

    p = h_norm**2 / mu.value

    inclination = np.arccos(h_vec[2] / h_norm)

    singularities = []
    if n_norm < tol:
        raan = 0.0
        singularities.append("equatorial_orbit")
    else:
        raan = _normalize_angle(np.arctan2(n_vec[1], n_vec[0]))

    if e < tol:
        argument_of_periapsis = 0.0
        singularities.append("circular_orbit")
    else:
        argument_of_periapsis = _normalize_angle(
            np.arctan2(np.dot(np.cross(n_vec, e_vec), h_vec) / h_norm, np.dot(n_vec, e_vec))
        )

    if e < tol:
        if n_norm < tol:
            true_anomaly = _normalize_angle(np.arctan2(r[1], r[0]))
        else:
            true_anomaly = _normalize_angle(
                np.arctan2(np.dot(np.cross(n_vec, r), h_vec) / h_norm, np.dot(n_vec, r))
            )
    else:
        true_anomaly = _normalize_angle(
            np.arctan2(np.dot(np.cross(e_vec, r), h_vec) / h_norm, np.dot(e_vec, r))
        )

    eccentric_anomaly = None
    mean_anomaly = 0.0
    if e < 1.0 - tol:
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(true_anomaly / 2.0),
            np.sqrt(1.0 + e) * np.cos(true_anomaly / 2.0),
        )
        mean_anomaly = float(eccentric_anomaly - e * np.sin(eccentric_anomaly))
    elif e > 1.0 + tol:
        eccentric_anomaly = 2.0 * np.arctanh(
            np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(true_anomaly / 2.0)
        )
        mean_anomaly = float(e * np.sinh(eccentric_anomaly) - eccentric_anomaly)
    else:
        d = np.tan(true_anomaly / 2.0)
        mean_anomaly = float(d + d**3 / 3.0)

    return OrbitalElements(
        semi_major_axis=float(a),
        eccentricity=float(e),
        inclination=float(inclination),
        raan=float(raan),
        argument_of_periapsis=float(argument_of_periapsis),
        true_anomaly=float(true_anomaly),
        eccentric_anomaly=float(eccentric_anomaly) if eccentric_anomaly is not None else None,
        mean_anomaly=float(mean_anomaly),
        semi_latus_rectum=float(p),
        singularities=tuple(singularities),
    )


def elements_to_state(
    body: CelestialBody,
    elements: OrbitalElements,
    time: Quantity,
    reference_frame: str = "inertial",
) -> OrbitState:
    """Convert orbital elements to a state vector at the given time."""
    mu = require_scalar_quantity(
        body.gravitational_parameter, "m^3/s^2", name="gravitational_parameter"
    )
    require_scalar_quantity(time, "s", name="time")

    a = elements.semi_major_axis
    e = elements.eccentricity
    i = elements.inclination
    raan = elements.raan
    argp = elements.argument_of_periapsis
    nu = elements.true_anomaly

    if np.isinf(a):
        p = elements.semi_latus_rectum
    else:
        p = a * (1.0 - e**2) if e < 1.0 else a * (e**2 - 1.0)

    r_pf = p / (1.0 + e * np.cos(nu))
    r_pqw = np.array([r_pf * np.cos(nu), r_pf * np.sin(nu), 0.0])
    v_pqw = np.array(
        [
            -np.sqrt(mu.value / p) * np.sin(nu),
            np.sqrt(mu.value / p) * (e + np.cos(nu)),
            0.0,
        ]
    )

    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_argp = np.cos(argp)
    sin_argp = np.sin(argp)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    rotation = np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
                -cos_raan * sin_i,
            ],
            [
                sin_argp * sin_i,
                cos_argp * sin_i,
                cos_i,
            ],
        ]
    )

    r_ijk = rotation @ r_pqw
    v_ijk = rotation @ v_pqw

    return OrbitState(
        position=Quantity(r_ijk, "m"),
        velocity=Quantity(v_ijk, "m/s"),
        time=time,
        reference_frame=reference_frame,
    )


__all__ = ["OrbitalElements", "state_to_elements", "elements_to_state"]
