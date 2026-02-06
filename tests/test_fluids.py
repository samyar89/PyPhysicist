import numpy as np

from PyPhysicist.fluids import (
    FluidProperties,
    FlowField,
    ControlSurface,
    ControlVolume,
    couette_flow,
    poiseuille_flow,
    reynolds_number,
)
from PyPhysicist.units import Quantity


def test_fluid_properties_dimensional_consistency():
    props = FluidProperties(
        density=Quantity(1000.0, "kg/m^3"),
        dynamic_viscosity=Quantity(1.0e-3, "kg/m/s"),
    )
    assert np.isclose(props.kinematic_viscosity_value(), 1.0e-6)


def test_dimensionless_reynolds_number():
    props = FluidProperties(density=1.2, dynamic_viscosity=1.8e-5)
    group = reynolds_number(props, velocity=5.0, length=0.5)
    assert np.isclose(group.value, 1.2 * 5.0 * 0.5 / 1.8e-5)


def test_couette_flow_residuals_near_zero():
    solution = couette_flow(velocity_top=2.0, height=1.0)
    residual = solution.validation()
    assert residual < 1e-6


def test_poiseuille_flow_residuals_near_zero():
    solution = poiseuille_flow(pressure_gradient=1.0, height=1.0, viscosity=1.0)
    residual = solution.validation()
    assert residual < 1e-6


def test_control_volume_mass_balance():
    flow = FlowField(velocity=lambda pts, _: np.array([1.0]))
    surfaces = [
        ControlSurface(centroid=np.array([0.0]), normal=np.array([-1.0]), area=1.0),
        ControlSurface(centroid=np.array([1.0]), normal=np.array([1.0]), area=1.0),
    ]
    volume = ControlVolume(surfaces=surfaces)
    props = FluidProperties(density=1.0, dynamic_viscosity=1.0)
    flux = volume.mass_flux(flow, props)
    assert np.isclose(flux, 0.0)
