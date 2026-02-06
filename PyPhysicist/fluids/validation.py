"""Validation helpers for physical sanity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ControlVolume, FlowField, FluidProperties
from .equations import continuity_equation, navier_stokes_equations


@dataclass
class Diagnostic:
    name: str
    value: float
    passed: bool
    details: str


def check_mass_conservation(flow: FlowField, properties: FluidProperties, tol: float = 1e-6) -> Diagnostic:
    residual = continuity_equation(flow, properties).evaluate(flow, properties)
    value = float(np.max(np.abs(residual)))
    return Diagnostic("MassConservation", value, value < tol, "Max continuity residual")


def check_momentum_conservation(flow: FlowField, properties: FluidProperties, tol: float = 1e-6) -> Diagnostic:
    residual = navier_stokes_equations(flow, properties).evaluate(flow, properties)
    value = float(np.max(np.abs(residual)))
    return Diagnostic("MomentumConservation", value, value < tol, "Max momentum residual")


def check_boundary_condition(flow: FlowField, points: np.ndarray, expected_velocity: np.ndarray, tol: float = 1e-6) -> Diagnostic:
    velocity = flow.evaluate_velocity(points)
    value = float(np.max(np.abs(np.asarray(velocity) - expected_velocity)))
    return Diagnostic("BoundaryCondition", value, value < tol, "Velocity boundary mismatch")


def check_positive_dissipation(flow: FlowField, properties: FluidProperties, tol: float = 0.0) -> Diagnostic:
    velocity = np.asarray(flow.evaluate_velocity(flow.grid))
    grad_u = np.gradient(velocity, axis=0)
    dissipation = np.sum(grad_u**2)
    value = float(dissipation)
    return Diagnostic("PositiveDissipation", value, value >= tol, "Viscous dissipation")


def check_control_volume_mass(control_volume: ControlVolume, flow: FlowField, properties: FluidProperties, tol: float = 1e-6) -> Diagnostic:
    flux = control_volume.mass_flux(flow, properties)
    return Diagnostic("ControlVolumeMass", float(abs(flux)), abs(flux) < tol, "Net mass flux")


__all__ = [
    "Diagnostic",
    "check_mass_conservation",
    "check_momentum_conservation",
    "check_boundary_condition",
    "check_positive_dissipation",
    "check_control_volume_mass",
]
