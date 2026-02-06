import numpy as np

from PyPhysicist.astrodynamics import (
    CelestialBody,
    OrbitState,
    OrbitalElements,
    elements_to_state,
    elliptic_kepler_orbit,
    kepler_third_law,
    state_to_elements,
    two_body_equation,
    validate_conservation,
    validate_keplerian_constraints,
)
from PyPhysicist.astrodynamics.solutions import circular_orbit
from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import parse_unit


def test_two_body_equation_dimensionality():
    earth = CelestialBody(
        gravitational_parameter=Quantity(3.986004418e14, "m^3/s^2"),
        mass=Quantity(5.972e24, "kg"),
        radius=Quantity(6.371e6, "m"),
    )
    state = OrbitState(
        position=Quantity(np.array([7.0e6, 0.0, 0.0]), "m"),
        velocity=Quantity(np.array([0.0, 7.5e3, 0.0]), "m/s"),
        time=Quantity(0.0, "s"),
    )
    eq = two_body_equation(earth, state)
    assert parse_unit(eq["expected_acceleration"].unit).dims == parse_unit("m/s^2").dims


def test_kepler_third_law_period():
    earth = CelestialBody(
        gravitational_parameter=Quantity(3.986004418e14, "m^3/s^2"),
        mass=Quantity(5.972e24, "kg"),
        radius=Quantity(6.371e6, "m"),
    )
    a = Quantity(7.0e6, "m")
    result = kepler_third_law(earth, a)
    expected = 2 * np.pi * np.sqrt(a.value**3 / earth.gravitational_parameter.value)
    assert np.isclose(result["period"].value, expected)


def test_energy_and_angular_momentum_conservation():
    earth = CelestialBody(
        gravitational_parameter=Quantity(3.986004418e14, "m^3/s^2"),
        mass=Quantity(5.972e24, "kg"),
        radius=Quantity(6.371e6, "m"),
    )
    radius = 7.0e6
    speed = np.sqrt(earth.gravitational_parameter.value / radius)
    state1 = OrbitState(
        position=Quantity(np.array([radius, 0.0, 0.0]), "m"),
        velocity=Quantity(np.array([0.0, speed, 0.0]), "m/s"),
        time=Quantity(0.0, "s"),
    )
    state2 = OrbitState(
        position=Quantity(np.array([0.0, radius, 0.0]), "m"),
        velocity=Quantity(np.array([-speed, 0.0, 0.0]), "m/s"),
        time=Quantity(100.0, "s"),
    )
    report = validate_conservation(earth, [state1, state2])
    assert report.checks["specific_energy"]["max_relative_deviation"] < 1e-12
    assert report.checks["angular_momentum"]["max_relative_deviation"] < 1e-12


def _angle_close(angle_a, angle_b, tol=1e-6):
    return np.isclose(np.mod(angle_a - angle_b, 2 * np.pi), 0.0, atol=tol) or np.isclose(
        np.mod(angle_b - angle_a, 2 * np.pi), 0.0, atol=tol
    )


def test_state_elements_conversion_round_trip():
    earth = CelestialBody(
        gravitational_parameter=Quantity(3.986004418e14, "m^3/s^2"),
        mass=Quantity(5.972e24, "kg"),
        radius=Quantity(6.371e6, "m"),
    )
    elements = OrbitalElements(
        semi_major_axis=10.0e6,
        eccentricity=0.1,
        inclination=0.3,
        raan=0.4,
        argument_of_periapsis=0.5,
        true_anomaly=1.0,
        eccentric_anomaly=0.9,
        mean_anomaly=0.8,
        semi_latus_rectum=10.0e6 * (1 - 0.1**2),
        singularities=(),
    )
    reconstructed = elements_to_state(
        earth,
        elements,
        time=Quantity(0.0, "s"),
    )
    round_trip = state_to_elements(earth, reconstructed)
    assert np.isclose(round_trip.semi_major_axis, elements.semi_major_axis, rtol=1e-6)
    assert np.isclose(round_trip.eccentricity, elements.eccentricity, rtol=1e-6)
    assert _angle_close(round_trip.inclination, elements.inclination, tol=1e-6)
    assert _angle_close(round_trip.raan, elements.raan, tol=1e-6)
    assert _angle_close(round_trip.argument_of_periapsis, elements.argument_of_periapsis, tol=1e-6)
    assert _angle_close(round_trip.true_anomaly, elements.true_anomaly, tol=1e-6)


def test_circular_and_elliptic_solutions():
    mu = Quantity(3.986004418e14, "m^3/s^2")
    radius = Quantity(7.0e6, "m")
    circular = circular_orbit(mu, radius)
    r0 = circular["r"](0.0)
    r1 = circular["r"](1000.0)
    assert np.isclose(r0, r1)

    a = Quantity(10.0e6, "m")
    e = 0.2
    elliptic = elliptic_kepler_orbit(mu, a, e)
    t = 500.0
    r = elliptic["r"](t)
    theta = elliptic["theta"](t)
    expected_r = a.value * (1 - e**2) / (1 + e * np.cos(theta))
    assert np.isclose(r, expected_r)


def test_validate_keplerian_constraints():
    earth = CelestialBody(
        gravitational_parameter=Quantity(3.986004418e14, "m^3/s^2"),
        mass=Quantity(5.972e24, "kg"),
        radius=Quantity(6.371e6, "m"),
    )
    radius = 7.0e6
    speed = np.sqrt(earth.gravitational_parameter.value / radius)
    state = OrbitState(
        position=Quantity(np.array([radius, 0.0, 0.0]), "m"),
        velocity=Quantity(np.array([0.0, speed, 0.0]), "m/s"),
        time=Quantity(0.0, "s"),
    )
    report = validate_keplerian_constraints(earth, state)
    assert abs(report.checks["vis_viva_residual"].value) < 1e-6
