"""Analytical solutions for canonical Keplerian orbits."""

from __future__ import annotations

import numpy as np

from PyPhysicist.units import Quantity
from ._utils import require_scalar_quantity


def _solve_kepler_equation(mean_anomaly: float, eccentricity: float) -> float:
    """Solve Kepler's equation E - e sin E = M for elliptical orbits."""
    e = eccentricity
    m = np.mod(mean_anomaly, 2 * np.pi)
    if e < 1e-8:
        return m
    e_anom = m if e < 0.8 else np.pi
    for _ in range(50):
        f = e_anom - e * np.sin(e_anom) - m
        f_prime = 1 - e * np.cos(e_anom)
        delta = -f / f_prime
        e_anom += delta
        if abs(delta) < 1e-12:
            break
    return float(e_anom)


def _solve_hyperbolic_kepler(mean_anomaly: float, eccentricity: float) -> float:
    """Solve hyperbolic Kepler equation e sinh H - H = M."""
    e = eccentricity
    m = mean_anomaly
    h_anom = np.arcsinh(m / e) if e > 1.0 else m
    for _ in range(50):
        f = e * np.sinh(h_anom) - h_anom - m
        f_prime = e * np.cosh(h_anom) - 1
        delta = -f / f_prime
        h_anom += delta
        if abs(delta) < 1e-12:
            break
    return float(h_anom)


def _solve_parabolic(mean_anomaly: float) -> float:
    """Solve Barker's equation M = D + D^3/3 for D."""
    d = mean_anomaly
    for _ in range(50):
        f = d + d**3 / 3.0 - mean_anomaly
        f_prime = 1 + d**2
        delta = -f / f_prime
        d += delta
        if abs(delta) < 1e-12:
            break
    return float(d)


def circular_orbit(mu: Quantity, radius: Quantity, theta0: float = 0.0) -> dict:
    """Return analytical expressions for a circular orbit."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    r_q = require_scalar_quantity(radius, "m", name="radius")
    n = np.sqrt(mu_q.value / r_q.value**3)

    def r_of_t(t: float) -> float:
        return float(r_q.value)

    def theta_of_t(t: float) -> float:
        return float(theta0 + n * t)

    return {
        "assumptions": "Circular orbit, e=0, planar motion.",
        "mean_motion": Quantity(n, "1/s"),
        "r": r_of_t,
        "theta": theta_of_t,
    }


def elliptic_kepler_orbit(
    mu: Quantity,
    semi_major_axis: Quantity,
    eccentricity: float,
    t0: float = 0.0,
    mean_anomaly0: float = 0.0,
) -> dict:
    """Return analytical expressions for an elliptic Keplerian orbit."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    a_q = require_scalar_quantity(semi_major_axis, "m", name="semi_major_axis")
    e = eccentricity
    n = np.sqrt(mu_q.value / a_q.value**3)

    def r_of_t(t: float) -> float:
        m = mean_anomaly0 + n * (t - t0)
        e_anom = _solve_kepler_equation(m, e)
        return float(a_q.value * (1 - e * np.cos(e_anom)))

    def theta_of_t(t: float) -> float:
        m = mean_anomaly0 + n * (t - t0)
        e_anom = _solve_kepler_equation(m, e)
        return float(
            2
            * np.arctan2(
                np.sqrt(1 + e) * np.sin(e_anom / 2),
                np.sqrt(1 - e) * np.cos(e_anom / 2),
            )
        )

    return {
        "assumptions": "Elliptic orbit, 0<e<1, two-body Keplerian motion.",
        "mean_motion": Quantity(n, "1/s"),
        "r": r_of_t,
        "theta": theta_of_t,
    }


def parabolic_escape(mu: Quantity, periapsis_distance: Quantity, t0: float = 0.0) -> dict:
    """Return analytical expressions for a parabolic escape trajectory."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    rp_q = require_scalar_quantity(periapsis_distance, "m", name="periapsis_distance")
    p = 2.0 * rp_q.value
    n = np.sqrt(mu_q.value / p**3)

    def r_of_t(t: float) -> float:
        m = n * (t - t0)
        d = _solve_parabolic(m)
        return float(p / (1 + d**2))

    def theta_of_t(t: float) -> float:
        m = n * (t - t0)
        d = _solve_parabolic(m)
        return float(2 * np.arctan(d))

    return {
        "assumptions": "Parabolic escape, e=1, two-body Keplerian motion.",
        "time_scale": Quantity(1 / n, "s"),
        "r": r_of_t,
        "theta": theta_of_t,
    }


def hyperbolic_flyby(
    mu: Quantity,
    semi_major_axis: Quantity,
    eccentricity: float,
    t0: float = 0.0,
    mean_anomaly0: float = 0.0,
) -> dict:
    """Return analytical expressions for a hyperbolic flyby orbit."""
    mu_q = require_scalar_quantity(mu, "m^3/s^2", name="gravitational_parameter")
    a_q = require_scalar_quantity(semi_major_axis, "m", name="semi_major_axis")
    e = eccentricity
    n = np.sqrt(mu_q.value / abs(a_q.value) ** 3)

    def r_of_t(t: float) -> float:
        m = mean_anomaly0 + n * (t - t0)
        h_anom = _solve_hyperbolic_kepler(m, e)
        return float(a_q.value * (e * np.cosh(h_anom) - 1))

    def theta_of_t(t: float) -> float:
        m = mean_anomaly0 + n * (t - t0)
        h_anom = _solve_hyperbolic_kepler(m, e)
        return float(
            2
            * np.arctan2(
                np.sqrt(e + 1) * np.sinh(h_anom / 2),
                np.sqrt(e - 1),
            )
        )

    return {
        "assumptions": "Hyperbolic flyby, e>1, two-body Keplerian motion.",
        "mean_motion": Quantity(n, "1/s"),
        "r": r_of_t,
        "theta": theta_of_t,
    }


__all__ = [
    "circular_orbit",
    "elliptic_kepler_orbit",
    "parabolic_escape",
    "hyperbolic_flyby",
]
