"""Operator form governing equations for fluid mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .core import FlowField, FluidProperties, _quantity_value


@dataclass
class ResidualOperator:
    """Residual operator for governing equations."""

    name: str
    symbolic_form: str
    evaluator: Callable[[FlowField, FluidProperties], Any]

    def evaluate(self, flow: FlowField, properties: FluidProperties) -> Any:
        return self.evaluator(flow, properties)


def continuity_equation(flow: FlowField, properties: FluidProperties, *, incompressible: bool = True) -> ResidualOperator:
    """Continuity equation residual operator."""

    if incompressible:
        symbolic = "∇·u"

        def evaluator(field: FlowField, _: FluidProperties) -> Any:
            velocity = field.evaluate_velocity(field.grid)
            return field.divergence(np.asarray(velocity))

    else:
        symbolic = "∂ρ/∂t + ∇·(ρu)"

        def evaluator(field: FlowField, props: FluidProperties) -> Any:
            velocity = np.asarray(field.evaluate_velocity(field.grid))
            rho = _quantity_value(props.density)
            return field.divergence(rho * velocity)

    return ResidualOperator("continuity", symbolic, evaluator)


def euler_equations(flow: FlowField, properties: FluidProperties) -> ResidualOperator:
    """Euler equations for inviscid flow."""

    symbolic = "∂u/∂t + (u·∇)u + (1/ρ)∇p"

    def evaluator(field: FlowField, props: FluidProperties) -> Any:
        velocity = np.asarray(field.evaluate_velocity(field.grid))
        pressure = np.asarray(field.evaluate_pressure(field.grid))
        grad_p = field.gradient(pressure)
        rho = _quantity_value(props.density)
        nonlinear = _convective_term(field, velocity)
        return nonlinear + grad_p / rho

    return ResidualOperator("euler", symbolic, evaluator)


def navier_stokes_equations(
    flow: FlowField,
    properties: FluidProperties,
    *,
    incompressible: bool = True,
) -> ResidualOperator:
    """Navier-Stokes equations residual operator."""

    symbolic = "∂u/∂t + (u·∇)u + (1/ρ)∇p - ν∇²u"

    def evaluator(field: FlowField, props: FluidProperties) -> Any:
        velocity = np.asarray(field.evaluate_velocity(field.grid))
        pressure = np.asarray(field.evaluate_pressure(field.grid))
        grad_p = field.gradient(pressure)
        rho = _quantity_value(props.density)
        nu = _quantity_value(props.kinematic_viscosity)
        nonlinear = _convective_term(field, velocity)
        viscous = _vector_laplacian(field, velocity)
        return nonlinear + grad_p / rho - nu * viscous

    return ResidualOperator("navier_stokes", symbolic, evaluator)


def vorticity_transport_equation(flow: FlowField, properties: FluidProperties) -> ResidualOperator:
    """Vorticity transport equation for incompressible flow."""

    symbolic = "∂ω/∂t + (u·∇)ω - (ω·∇)u - ν∇²ω"

    def evaluator(field: FlowField, props: FluidProperties) -> Any:
        velocity = np.asarray(field.evaluate_velocity(field.grid))
        omega = _curl(field, velocity)
        nu = _quantity_value(props.kinematic_viscosity)
        convective = _convective_term(field, omega)
        stretch = _stretching_term(field, velocity, omega)
        viscous = _vector_laplacian(field, omega)
        return convective - stretch - nu * viscous

    return ResidualOperator("vorticity_transport", symbolic, evaluator)


def energy_equation(flow: FlowField, properties: FluidProperties, *, conductivity: float) -> ResidualOperator:
    """Classical energy equation (internal energy form)."""

    symbolic = "ρ c_p (∂T/∂t + u·∇T) - k∇²T"

    def evaluator(field: FlowField, props: FluidProperties) -> Any:
        temperature = np.asarray(field.evaluate_temperature(field.grid))
        velocity = np.asarray(field.evaluate_velocity(field.grid))
        rho = _quantity_value(props.density)
        convective = _convective_scalar(field, velocity, temperature)
        diffusion = field.laplacian(temperature)
        return rho * convective - conductivity * diffusion

    return ResidualOperator("energy", symbolic, evaluator)


def _convective_term(field: FlowField, velocity: np.ndarray) -> np.ndarray:
    grad_u = _velocity_gradient(field, velocity)
    return np.einsum("...i,...ij->...j", velocity, grad_u)


def _convective_scalar(field: FlowField, velocity: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    grad_scalar = field.gradient(scalar)
    return np.einsum("...i,...i->...", velocity, grad_scalar)


def _velocity_gradient(field: FlowField, velocity: np.ndarray) -> np.ndarray:
    gradients = []
    for i in range(velocity.shape[-1]):
        gradients.append(field.gradient(velocity[..., i]))
    return np.stack(gradients, axis=-2)


def _vector_laplacian(field: FlowField, vector: np.ndarray) -> np.ndarray:
    components = [field.laplacian(vector[..., i]) for i in range(vector.shape[-1])]
    return np.stack(components, axis=-1)


def _curl(field: FlowField, velocity: np.ndarray) -> np.ndarray:
    if velocity.shape[-1] == 2:
        dvdx = field.gradient(velocity[..., 1])[..., 0]
        dudy = field.gradient(velocity[..., 0])[..., 1]
        return (dvdx - dudy)[..., None]
    grad = _velocity_gradient(field, velocity)
    curl = np.zeros_like(velocity)
    curl[..., 0] = grad[..., 2, 1] - grad[..., 1, 2]
    curl[..., 1] = grad[..., 0, 2] - grad[..., 2, 0]
    curl[..., 2] = grad[..., 1, 0] - grad[..., 0, 1]
    return curl


def _stretching_term(field: FlowField, velocity: np.ndarray, omega: np.ndarray) -> np.ndarray:
    grad_u = _velocity_gradient(field, velocity)
    return np.einsum("...i,...ij->...j", omega, grad_u)


__all__ = [
    "ResidualOperator",
    "continuity_equation",
    "euler_equations",
    "navier_stokes_equations",
    "vorticity_transport_equation",
    "energy_equation",
]
