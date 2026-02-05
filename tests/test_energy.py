import pytest

from PyPhysicist.mechanics import energy as mechanics_energy


def test_kinetic_energy():
    assert mechanics_energy.kinetic_energy(2.0, 3.0) == 9.0


def test_gravitational_potential_energy():
    assert mechanics_energy.gravitational_potential_energy(2.0, 9.81, 5.0) == pytest.approx(98.1)


def test_mechanical_energy():
    assert mechanics_energy.mechanical_energy(10.0, 15.5) == 25.5


def test_elastic_potential_energy():
    assert mechanics_energy.elastic_potential_energy(100.0, 0.2) == pytest.approx(2.0)
