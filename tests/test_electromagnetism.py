import pytest

from PyPhysicist import electromagnetism


def test_coulomb_force():
    force = electromagnetism.coulomb_force(1e-6, 2e-6, 0.05)
    assert force == pytest.approx(7.19004143384)


def test_electric_field():
    assert electromagnetism.electric_field(10.0, 2.0) == 5.0


def test_capacitance():
    assert electromagnetism.capacitance(5.0, 2.0) == 2.5


def test_resistance_series():
    assert electromagnetism.resistance_series(1.0, 2.0, 3.0) == 6.0


def test_resistance_parallel():
    assert electromagnetism.resistance_parallel(6.0, 3.0) == 2.0
