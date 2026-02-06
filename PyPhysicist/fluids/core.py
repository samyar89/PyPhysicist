"""Core abstractions for analytical fluid mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from PyPhysicist.units import Quantity, UnitError
from PyPhysicist.units.conversion import parse_unit


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, Quantity):
        return np.asarray(value.value)
    return np.asarray(value)


def _validate_unit(value: Any, expected_unit: str, label: str) -> None:
    if isinstance(value, Quantity):
        expected = parse_unit(expected_unit)
        actual = parse_unit(value.unit)
        if expected.dims != actual.dims:
            raise UnitError(
                f"{label} expects units compatible with '{expected_unit}', got '{value.unit}'."
            )


def _quantity_value(value: Any) -> Any:
    return value.value if isinstance(value, Quantity) else value


@dataclass
class FluidProperties:
    """Physical properties of a fluid.

    Attributes:
        density: Mass density (rho).
        dynamic_viscosity: Dynamic viscosity (mu).
        kinematic_viscosity: Kinematic viscosity (nu).
        compressibility: Compressibility (beta).
        equation_of_state: Callable that returns density from (pressure, temperature).
    """

    density: Any
    dynamic_viscosity: Any | None = None
    kinematic_viscosity: Any | None = None
    compressibility: Any | None = None
    equation_of_state: Optional[Callable[[Any, Any], Any]] = None

    def __post_init__(self) -> None:
        _validate_unit(self.density, "kg/m^3", "density")
        if self.dynamic_viscosity is not None:
            _validate_unit(self.dynamic_viscosity, "kg/m/s", "dynamic_viscosity")
        if self.kinematic_viscosity is not None:
            _validate_unit(self.kinematic_viscosity, "m^2/s", "kinematic_viscosity")
        if self.dynamic_viscosity is None and self.kinematic_viscosity is not None:
            self.dynamic_viscosity = _quantity_value(self.kinematic_viscosity) * _quantity_value(
                self.density
            )
        if self.kinematic_viscosity is None and self.dynamic_viscosity is not None:
            self.kinematic_viscosity = _quantity_value(self.dynamic_viscosity) / _quantity_value(
                self.density
            )

    def density_value(self) -> Any:
        return _quantity_value(self.density)

    def dynamic_viscosity_value(self) -> Any:
        return _quantity_value(self.dynamic_viscosity)

    def kinematic_viscosity_value(self) -> Any:
        return _quantity_value(self.kinematic_viscosity)

    def symbolic(self) -> dict[str, Any]:
        return {
            "density": self.density,
            "dynamic_viscosity": self.dynamic_viscosity,
            "kinematic_viscosity": self.kinematic_viscosity,
            "compressibility": self.compressibility,
            "equation_of_state": self.equation_of_state,
        }


@dataclass
class FlowField:
    """Flow field abstraction supporting symbolic and numeric access."""

    velocity: Any
    pressure: Any | None = None
    temperature: Any | None = None
    grid: tuple[np.ndarray, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def _evaluate(self, field: Any, points: Any, time: float | None) -> Any:
        if field is None:
            return None
        if callable(field):
            return field(points, time)
        return _as_array(field)

    def evaluate_velocity(self, points: Any, time: float | None = None) -> Any:
        return self._evaluate(self.velocity, points, time)

    def evaluate_pressure(self, points: Any, time: float | None = None) -> Any:
        return self._evaluate(self.pressure, points, time)

    def evaluate_temperature(self, points: Any, time: float | None = None) -> Any:
        return self._evaluate(self.temperature, points, time)

    def symbolic(self) -> dict[str, Any]:
        return {
            "velocity": self.velocity,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "grid": self.grid,
            "metadata": self.metadata,
        }

    def _grid_spacing(self) -> tuple[np.ndarray, ...]:
        if self.grid is None:
            raise ValueError("FlowField requires a grid for spatial derivatives.")
        spacing = []
        for axis in self.grid:
            axis = np.asarray(axis)
            if axis.ndim != 1:
                raise ValueError("Grid axes must be 1D arrays.")
            spacing.append(np.gradient(axis))
        return tuple(spacing)

    def divergence(self, vector_field: np.ndarray) -> np.ndarray:
        spacing = self._grid_spacing()
        if vector_field.shape[-1] != len(spacing):
            raise ValueError("Vector field dimension does not match grid dimension.")
        components = [vector_field[..., i] for i in range(vector_field.shape[-1])]
        grads = [np.gradient(comp, *spacing, edge_order=2) for comp in components]
        return sum(grad[i] for i, grad in enumerate(grads))

    def gradient(self, scalar_field: np.ndarray) -> np.ndarray:
        spacing = self._grid_spacing()
        grads = np.gradient(scalar_field, *spacing, edge_order=2)
        return np.stack(grads, axis=-1)

    def laplacian(self, scalar_field: np.ndarray) -> np.ndarray:
        spacing = self._grid_spacing()
        grads = np.gradient(scalar_field, *spacing, edge_order=2)
        second = [
            np.gradient(grad, *spacing, edge_order=2)[i]
            for i, grad in enumerate(grads)
        ]
        return sum(second)


@dataclass
class ControlSurface:
    """Simple control surface element with area and normal."""

    centroid: np.ndarray
    normal: np.ndarray
    area: float


@dataclass
class ControlVolume:
    """Control volume defined by surface elements."""

    surfaces: Iterable[ControlSurface]

    def mass_flux(self, flow: FlowField, properties: FluidProperties, time: float | None = None) -> float:
        rho = _quantity_value(properties.density)
        flux = 0.0
        for surface in self.surfaces:
            velocity = flow.evaluate_velocity(surface.centroid, time)
            flux += rho * float(np.dot(velocity, surface.normal)) * surface.area
        return float(flux)

    def momentum_flux(
        self, flow: FlowField, properties: FluidProperties, time: float | None = None
    ) -> np.ndarray:
        rho = _quantity_value(properties.density)
        total = None
        for surface in self.surfaces:
            velocity = flow.evaluate_velocity(surface.centroid, time)
            pressure = flow.evaluate_pressure(surface.centroid, time) or 0.0
            flux = rho * velocity * float(np.dot(velocity, surface.normal))
            flux = flux + pressure * surface.normal
            contribution = flux * surface.area
            total = contribution if total is None else total + contribution
        if total is None:
            raise ValueError(\"ControlVolume has no surfaces defined.\")
        return total

    def energy_flux(
        self,
        flow: FlowField,
        properties: FluidProperties,
        specific_energy: Callable[[np.ndarray], np.ndarray],
        time: float | None = None,
    ) -> float:
        rho = _quantity_value(properties.density)
        total = 0.0
        for surface in self.surfaces:
            velocity = flow.evaluate_velocity(surface.centroid, time)
            energy = specific_energy(surface.centroid)
            total += rho * energy * float(np.dot(velocity, surface.normal)) * surface.area
        return float(total)


__all__ = ["FluidProperties", "FlowField", "ControlSurface", "ControlVolume"]
