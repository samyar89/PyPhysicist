import pytest

import thermodynamics


def test_ideal_gas_pressure():
    assert thermodynamics.ideal_gas_pressure(1.0, 8.314, 300.0, 0.024) == pytest.approx(103925.0)


def test_heat_capacity():
    assert thermodynamics.heat_capacity(4200.0, 2.0, 10.0) == 210.0


def test_entropy_change():
    assert thermodynamics.entropy_change(500.0, 250.0) == 2.0
