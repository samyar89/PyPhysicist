import pytest

import PyPhysicist as pp


def test_velocity_unit_conversion():
    distance = pp.Quantity(1.0, "km")
    time = pp.Quantity(2.0, "s")
    result = pp.mechanics.velocity(distance, time)
    assert isinstance(result, pp.Quantity)
    assert result.unit == "m/s"
    assert result.value == pytest.approx(500.0)


def test_unit_error_on_incompatible_units():
    with pytest.raises(pp.UnitError):
        pp.mechanics.velocity(pp.Quantity(1.0, "kg"), pp.Quantity(1.0, "s"))


def test_heat_capacity_delta_temperature():
    heat = pp.Quantity(1000.0, "J")
    mass = pp.Quantity(1.0, "kg")
    delta_t = pp.Quantity(1.0, "degC")
    result = pp.thermodynamics.heat_capacity(heat, mass, delta_t)
    assert result.value == pytest.approx(1000.0)
    assert result.unit == "J/kg*K"
