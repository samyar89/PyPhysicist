import pytest

from PyPhysicist import waves_optics


def test_frequency():
    assert waves_optics.frequency(340.0, 0.34) == pytest.approx(1000.0)


def test_wavelength():
    assert waves_optics.wavelength(300.0, 150.0) == 2.0


def test_wave_power():
    assert waves_optics.wave_power(120.0, 4.0) == 30.0


def test_refractive_index():
    assert waves_optics.refractive_index(300_000_000, 200_000_000) == 1.5
