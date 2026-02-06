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


def test_quantity_arithmetic_with_units():
    force = pp.Quantity(575.0, "N")
    mass = pp.Quantity(10011.0, "Mg")
    acceleration = force / mass
    assert isinstance(acceleration, pp.Quantity)
    assert acceleration.unit == "m/s^2"
    assert acceleration.value == pytest.approx(575.0 / (10011.0 * 1_000.0))


def test_quantity_addition_with_unit_conversion():
    left = pp.Quantity(1.0, "m")
    right = pp.Quantity(50.0, "cm")
    result = left + right
    assert result.unit == "m"
    assert result.value == pytest.approx(1.5)
