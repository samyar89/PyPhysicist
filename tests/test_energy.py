import pytest

import Energy


def test_kinetic_energy():
    assert Energy.KINETIC_ENERGY(2.0, 3.0) == 9.0


def test_gravitational_potential_energy():
    assert Energy.GRAVITATIONAL_POTENTIAL_ENERGY(2.0, 9.81, 5.0) == pytest.approx(98.1)


def test_mechanical_energy():
    assert Energy.MECHANICAL_ENERGY(10.0, 15.5) == 25.5


def test_elastic_potential_energy():
    assert Energy.ELASTIC_POTENTIAL_ENERGY(100.0, 0.2) == pytest.approx(2.0)
