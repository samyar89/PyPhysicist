import pytest

from PyPhysicist import relativity


def test_time_dilation():
    dilated = relativity.time_dilation(10.0, 0.8, 1.0)
    assert dilated == pytest.approx(16.6666666667)


def test_length_contraction():
    contracted = relativity.length_contraction(10.0, 0.6, 1.0)
    assert contracted == pytest.approx(8.0)


def test_relativistic_energy():
    assert relativity.relativistic_energy(2.0, 3.0) == 18.0
