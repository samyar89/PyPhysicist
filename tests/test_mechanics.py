import pytest

import mechanics


def test_momentum():
    assert mechanics.momentum(2.0, 3.5) == 7.0


def test_newton_second_law():
    assert mechanics.newton_second_law(4.0, 2.5) == 10.0


def test_spring_potential_energy():
    assert mechanics.spring_potential_energy(200.0, 0.1) == pytest.approx(1.0)


def test_centripetal_acceleration():
    assert mechanics.centripetal_acceleration(4.0, 2.0) == 8.0


def test_centripetal_force():
    assert mechanics.centripetal_force(3.0, 4.0, 2.0) == 24.0
