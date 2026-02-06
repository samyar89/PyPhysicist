"""Canonical analytical solutions for classical flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .core import FlowField, FluidProperties
from .equations import navier_stokes_equations


@dataclass
class AnalyticalSolution:
    name: str
    assumptions: str
    flow: FlowField
    validation: Callable[[], float]


def couette_flow(velocity_top: float, height: float) -> AnalyticalSolution:
    """Plane Couette flow between plates y=0 and y=H."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        y = _extract_coordinate(points, 0)
        u = velocity_top * y / height
        return np.stack((u,), axis=-1)

    def pressure(points: np.ndarray, _: float | None = None) -> np.ndarray:
        y = _extract_coordinate(points, 0)
        return np.zeros_like(y)

    flow = FlowField(velocity=velocity, pressure=pressure)

    def validation() -> float:
        y = np.linspace(0, height, 64)
        grid = (y,)
        flow.grid = grid
        residual = navier_stokes_equations(flow, FluidProperties(density=1.0, dynamic_viscosity=1.0))
        value = residual.evaluate(flow, FluidProperties(density=1.0, dynamic_viscosity=1.0))
        return float(np.max(np.abs(value)))

    return AnalyticalSolution(
        name="Couette",
        assumptions="Steady, incompressible, laminar, no pressure gradient",
        flow=flow,
        validation=validation,
    )


def poiseuille_flow(pressure_gradient: float, height: float, viscosity: float) -> AnalyticalSolution:
    """Plane Poiseuille flow between plates y=0 and y=H."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        y = _extract_coordinate(points, 0)
        u = (pressure_gradient / (2 * viscosity)) * y * (height - y)
        return np.stack((u,), axis=-1)

    flow = FlowField(velocity=velocity, pressure=lambda pts, _: pressure_gradient * _extract_coordinate(pts, 0))

    def validation() -> float:
        y = np.linspace(0, height, 64)
        grid = (y,)
        flow.grid = grid
        props = FluidProperties(density=1.0, dynamic_viscosity=viscosity)
        residual = navier_stokes_equations(flow, props)
        value = residual.evaluate(flow, props)
        return float(np.max(np.abs(value)))

    return AnalyticalSolution(
        name="Poiseuille",
        assumptions="Steady, incompressible, laminar, constant pressure gradient",
        flow=flow,
        validation=validation,
    )


def hagen_poiseuille_flow(pressure_gradient: float, radius: float, viscosity: float) -> AnalyticalSolution:
    """Hagen-Poiseuille pipe flow in cylindrical coordinates."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        r = _extract_coordinate(points, 0)
        u = (pressure_gradient / (4 * viscosity)) * (radius**2 - r**2)
        return np.stack((u,), axis=-1)

    flow = FlowField(velocity=velocity, pressure=lambda pts, _: pressure_gradient * _extract_coordinate(pts, 0))

    def validation() -> float:
        r = np.linspace(0, radius, 64)
        grid = (r,)
        flow.grid = grid
        props = FluidProperties(density=1.0, dynamic_viscosity=viscosity)
        residual = navier_stokes_equations(flow, props)
        value = residual.evaluate(flow, props)
        return float(np.max(np.abs(value)))

    return AnalyticalSolution(
        name="Hagen-Poiseuille",
        assumptions="Steady, incompressible, laminar, axisymmetric",
        flow=flow,
        validation=validation,
    )


def stokes_first_problem(plate_velocity: float, kinematic_viscosity: float, time: float) -> AnalyticalSolution:
    """Stokes' first problem for impulsively started plate."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        y = _extract_coordinate(points, 0)
        eta = y / (2 * np.sqrt(kinematic_viscosity * time))
        u = plate_velocity * np.erfc(eta)
        return np.stack((u,), axis=-1)

    flow = FlowField(velocity=velocity, pressure=lambda pts, _: np.zeros_like(_extract_coordinate(pts, 0)))

    def validation() -> float:
        y = np.linspace(0, 5.0, 128)
        flow.grid = (y,)
        props = FluidProperties(density=1.0, kinematic_viscosity=kinematic_viscosity)
        residual = navier_stokes_equations(flow, props)
        value = residual.evaluate(flow, props)
        return float(np.max(np.abs(value)))

    return AnalyticalSolution(
        name="StokesFirstProblem",
        assumptions="Unsteady, incompressible, semi-infinite domain",
        flow=flow,
        validation=validation,
    )


def potential_source(strength: float) -> FlowField:
    """Potential flow source element."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        x = _extract_coordinate(points, 0)
        y = _extract_coordinate(points, 1)
        r2 = x**2 + y**2
        u = strength / (2 * np.pi) * x / r2
        v = strength / (2 * np.pi) * y / r2
        return np.stack((u, v), axis=-1)

    return FlowField(velocity=velocity, pressure=lambda pts, _: np.zeros_like(pts[0]))


def potential_sink(strength: float) -> FlowField:
    return potential_source(-strength)


def potential_vortex(circulation: float) -> FlowField:
    """Potential flow vortex element."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        x = _extract_coordinate(points, 0)
        y = _extract_coordinate(points, 1)
        r2 = x**2 + y**2
        u = -circulation / (2 * np.pi) * y / r2
        v = circulation / (2 * np.pi) * x / r2
        return np.stack((u, v), axis=-1)

    return FlowField(velocity=velocity, pressure=lambda pts, _: np.zeros_like(pts[0]))


def potential_doublet(strength: float) -> FlowField:
    """Potential flow doublet element."""

    def velocity(points: np.ndarray, _: float | None = None) -> np.ndarray:
        x = _extract_coordinate(points, 0)
        y = _extract_coordinate(points, 1)
        r2 = x**2 + y**2
        u = -strength / (2 * np.pi) * (x**2 - y**2) / r2**2
        v = -strength / (2 * np.pi) * (2 * x * y) / r2**2
        return np.stack((u, v), axis=-1)

    return FlowField(velocity=velocity, pressure=lambda pts, _: np.zeros_like(pts[0]))


def blasius_boundary_layer(eta_max: float = 8.0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Numerical Blasius solution via shooting method."""

    def integrate(fpp0: float) -> tuple[np.ndarray, np.ndarray]:
        eta = np.linspace(0.0, eta_max, n)
        h = eta[1] - eta[0]
        f = np.zeros((n, 3))
        f[0] = [0.0, 0.0, fpp0]
        for i in range(n - 1):
            k1 = _blasius_rhs(f[i])
            k2 = _blasius_rhs(f[i] + 0.5 * h * k1)
            k3 = _blasius_rhs(f[i] + 0.5 * h * k2)
            k4 = _blasius_rhs(f[i] + h * k3)
            f[i + 1] = f[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return eta, f

    target = 1.0
    fpp0 = 0.332
    for _ in range(12):
        eta, f = integrate(fpp0)
        error = f[-1, 1] - target
        fpp0 -= error * 0.1
    return eta, f[:, 1]


def _blasius_rhs(state: np.ndarray) -> np.ndarray:
    f, fp, fpp = state
    return np.array([fp, fpp, -0.5 * f * fpp])


def _extract_coordinate(points: np.ndarray | tuple[np.ndarray, ...], index: int) -> np.ndarray:
    if isinstance(points, tuple):
        return np.asarray(points[index])
    points = np.asarray(points)
    if points.ndim == 1:
        return points
    return points[..., index]


__all__ = [
    "AnalyticalSolution",
    "couette_flow",
    "poiseuille_flow",
    "hagen_poiseuille_flow",
    "stokes_first_problem",
    "potential_source",
    "potential_sink",
    "potential_vortex",
    "potential_doublet",
    "blasius_boundary_layer",
]
