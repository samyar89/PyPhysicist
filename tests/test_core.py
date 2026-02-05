import pytest

import PyPhysicist as pp


def test_voltage():
    assert pp.V(2.0, 5.0) == 10.0


def test_current():
    assert pp.I(10.0, 5.0) == 2.0


def test_resistance():
    assert pp.R(10.0, 2.0) == 5.0


def test_velocity():
    assert pp.Velocity(100.0, 4.0) == 25.0


def test_force():
    assert pp.F(5.0, 2.0) == 10.0


def test_weight():
    assert pp.Weight(3.0, 9.8) == pytest.approx(29.4)


def test_work():
    assert pp.Work(20.0, 3.0) == 60.0


def test_schwarzschild_radius():
    radius = pp.Schwarzschild_radius(5.0)
    expected = (2 * 6.67430e-11 * 5.0) / (299_792_458 ** 2)
    assert radius == pytest.approx(expected)
