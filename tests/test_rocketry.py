from math import log

import pytest

import PyPhysicist as pp


def test_tsiolkovsky_dimensionality():
    rocket = pp.rocketry.RocketBody(
        dry_mass=pp.Quantity(100.0, "kg"),
        propellant_mass=pp.Quantity(400.0, "kg"),
        mass_flow_rate=pp.Quantity(5.0, "kg/s"),
        reference_area=pp.Quantity(1.0, "m^2"),
    )
    propulsion = pp.rocketry.PropulsionModel(
        exhaust_velocity=pp.Quantity(3000.0, "m/s")
    )
    delta_v = pp.Quantity(2000.0, "m/s")
    residual = pp.rocketry.tsiolkovsky_equation(
        rocket, propulsion, delta_v=delta_v
    ).evaluate()
    assert isinstance(residual, pp.Quantity)
    assert residual.unit == "m/s"


def test_tsiolkovsky_limit():
    rocket = pp.rocketry.RocketBody(
        dry_mass=100.0,
        propellant_mass=400.0,
        mass_flow_rate=5.0,
        reference_area=1.0,
    )
    propulsion = pp.rocketry.PropulsionModel(exhaust_velocity=3000.0)
    delta_v = 3000.0 * log(5.0)
    residual = pp.rocketry.tsiolkovsky_equation(
        rocket, propulsion, delta_v=delta_v
    ).evaluate()
    assert residual == pytest.approx(0.0)


def test_stage_mass_conservation():
    stages = [
        pp.rocketry.Stage(dry_mass=100.0, propellant_mass=300.0),
        pp.rocketry.Stage(dry_mass=50.0, propellant_mass=150.0),
    ]
    report = pp.rocketry.validate_stage_mass_conservation(stages, payload_mass=25.0)
    assert report.passed


def test_meshchersky_residual():
    rocket = pp.rocketry.RocketBody(
        dry_mass=100.0,
        propellant_mass=400.0,
        mass_flow_rate=5.0,
        reference_area=1.0,
    )
    propulsion = pp.rocketry.PropulsionModel(exhaust_velocity=3000.0)
    state = pp.rocketry.FlightState(
        position=0.0,
        velocity=0.0,
        acceleration=15.0,
        mass=1000.0,
        time=0.0,
    )
    residual = pp.rocketry.meshchersky_equation(
        rocket, propulsion, state=state, external_force=0.0
    ).evaluate()
    assert residual == pytest.approx(0.0)


def test_dimensionless_groups():
    ratio = pp.rocketry.mass_ratio(pp.Quantity(500.0, "kg"), pp.Quantity(100.0, "kg"))
    twr = pp.rocketry.thrust_to_weight(
        pp.Quantity(10000.0, "N"), pp.Quantity(1000.0, "kg"), pp.Quantity(10.0, "m/s^2")
    )
    beta = pp.rocketry.ballistic_coefficient(
        pp.Quantity(1000.0, "kg"), 0.5, pp.Quantity(2.0, "m^2")
    )
    assert ratio.unit == "1"
    assert ratio.value == pytest.approx(5.0)
    assert twr.unit == "1"
    assert twr.value == pytest.approx(1.0)
    assert beta.unit == "kg/m^2"
    assert beta.value == pytest.approx(1000.0)
