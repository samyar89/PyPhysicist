import pytest

import PyPhysicist as pp


def test_voltage():
    assert pp.voltage(2.0, 5.0) == 10.0


def test_voltage_alias():
    assert pp.V(2.0, 5.0) == 10.0


def test_current():
    assert pp.current(10.0, 5.0) == 2.0


def test_current_alias():
    assert pp.I(10.0, 5.0) == 2.0


def test_resistance():
    assert pp.resistance(10.0, 2.0) == 5.0


def test_resistance_alias():
    assert pp.R(10.0, 2.0) == 5.0


def test_velocity():
    assert pp.velocity(100.0, 4.0) == 25.0


def test_velocity_alias():
    assert pp.Velocity(100.0, 4.0) == 25.0


def test_force():
    assert pp.force(5.0, 2.0) == 10.0


def test_force_alias():
    assert pp.F(5.0, 2.0) == 10.0


def test_weight():
    assert pp.weight(3.0, 9.8) == pytest.approx(29.4)


def test_weight_alias():
    assert pp.Weight(3.0, 9.8) == pytest.approx(29.4)


def test_work():
    assert pp.work(20.0, 3.0) == 60.0


def test_work_alias():
    assert pp.Work(20.0, 3.0) == 60.0


def test_schwarzschild_radius():
    radius = pp.schwarzschild_radius(5.0)
    expected = (2 * 6.67430e-11 * 5.0) / (299_792_458 ** 2)
    assert radius == pytest.approx(expected)


def test_schwarzschild_radius_alias():
    radius = pp.Schwarzschild_radius(5.0)
    expected = (2 * 6.67430e-11 * 5.0) / (299_792_458 ** 2)
    assert radius == pytest.approx(expected)
