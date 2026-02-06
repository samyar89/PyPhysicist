"""Core rocket physics abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from PyPhysicist.constants import STANDARD_GRAVITY
from PyPhysicist.units import Quantity, UnitError
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


def _call_with_optional_state(func: Callable[..., Any], t: float, state: Any) -> Any:
    try:
        return func(t, state)
    except TypeError:
        return func(t)


@dataclass(frozen=True)
class RocketBody:
    """Physical description of a rocket body.

    Args:
        dry_mass: Mass of structure without propellant.
        propellant_mass: Mass of propellant available.
        mass_flow_rate: Propellant mass flow rate (positive for expulsion).
        reference_area: Reference area for drag/lift.
        staging_info: Optional metadata for staging assumptions.
        assumptions: Explicit assumptions used to interpret the body model.
    """

    dry_mass: Quantity | float
    propellant_mass: Quantity | float
    mass_flow_rate: Quantity | float | Callable[..., Any] | str
    reference_area: Quantity | float
    staging_info: Optional[Dict[str, Any]] = None
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def total_mass(self) -> Quantity | float:
        dry_mass_value, dry_unit = coerce_value(self.dry_mass, "kg", name="dry_mass")
        prop_value, _ = coerce_value(self.propellant_mass, "kg", name="propellant_mass")
        total = dry_mass_value + prop_value
        return wrap_quantity(total, "kg", self.dry_mass, self.propellant_mass)

    def mass_rate(self, t: float, state: Any = None) -> Quantity | float | str:
        if isinstance(self.mass_flow_rate, str):
            return self.mass_flow_rate
        if callable(self.mass_flow_rate):
            return _call_with_optional_state(self.mass_flow_rate, t, state)
        rate_value, _ = coerce_value(
            self.mass_flow_rate, "kg/s", name="mass_flow_rate"
        )
        return wrap_quantity(rate_value, "kg/s", self.mass_flow_rate)

    def mass(self, t: float, t0: float = 0.0, state: Any = None) -> Quantity | float | str:
        if isinstance(self.mass_flow_rate, str):
            return f"m(t) = m0 - (mass_flow_rate)*({t} - {t0})"
        m0_value, _ = coerce_value(self.total_mass(), "kg", name="total_mass")
        dry_value, _ = coerce_value(self.dry_mass, "kg", name="dry_mass")
        mdot = self.mass_rate(t0, state)
        if isinstance(mdot, str):
            return f"m(t) = m0 - ({mdot})*({t} - {t0})"
        mdot_value, _ = coerce_value(mdot, "kg/s", name="mass_flow_rate")
        mass_value = m0_value - mdot_value * (t - t0)
        mass_value = np.maximum(mass_value, dry_value)
        return wrap_quantity(mass_value, "kg", self.dry_mass, self.propellant_mass, mdot)


@dataclass(frozen=True)
class PropulsionModel:
    """Propulsion model for thrust and exhaust characteristics."""

    exhaust_velocity: Quantity | float | None = None
    specific_impulse: Quantity | float | None = None
    thrust_model: Quantity | float | Callable[..., Any] | None = None
    chamber_pressure: Quantity | float | None = None
    ambient_pressure: Quantity | float | None = None
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def effective_exhaust_velocity(self) -> Quantity | float:
        if self.exhaust_velocity is not None:
            ve_value, _ = coerce_value(self.exhaust_velocity, "m/s", name="exhaust_velocity")
            return wrap_quantity(ve_value, "m/s", self.exhaust_velocity)
        if self.specific_impulse is None:
            raise UnitError("Either exhaust_velocity or specific_impulse must be provided.")
        isp_value, _ = coerce_value(self.specific_impulse, "s", name="specific_impulse")
        ve_value = isp_value * STANDARD_GRAVITY
        return wrap_quantity(ve_value, "m/s", self.specific_impulse)

    def thrust(self, t: float, state: Any = None, *, mass_flow_rate: Any = None, reference_area: Any = None) -> Quantity | float:
        if isinstance(self.thrust_model, str):
            raise UnitError("Symbolic thrust models must be inspected, not evaluated numerically.")
        if callable(self.thrust_model):
            return _call_with_optional_state(self.thrust_model, t, state)
        if self.thrust_model is not None:
            thrust_value, _ = coerce_value(self.thrust_model, "N", name="thrust")
            return wrap_quantity(thrust_value, "N", self.thrust_model)
        if mass_flow_rate is None:
            raise UnitError("mass_flow_rate is required to compute thrust.")
        mdot_value, _ = coerce_value(mass_flow_rate, "kg/s", name="mass_flow_rate")
        ve_value, _ = coerce_value(self.effective_exhaust_velocity(), "m/s", name="exhaust_velocity")
        thrust_value = mdot_value * ve_value
        if self.chamber_pressure is not None and reference_area is not None:
            chamber_value, _ = coerce_value(self.chamber_pressure, "Pa", name="chamber_pressure")
            ambient_value, _ = coerce_value(self.ambient_pressure or 0.0, "Pa", name="ambient_pressure")
            area_value, _ = coerce_value(reference_area, "m^2", name="reference_area")
            thrust_value = thrust_value + (chamber_value - ambient_value) * area_value
        return wrap_quantity(thrust_value, "N", mass_flow_rate, self.exhaust_velocity, self.specific_impulse)


@dataclass(frozen=True)
class FlightState:
    """State of flight for evaluating equations of motion."""

    position: Any
    velocity: Any
    acceleration: Any
    mass: Quantity | float
    time: Quantity | float
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "mass": self.mass,
            "time": self.time,
            "assumptions": self.assumptions,
        }


__all__ = ["RocketBody", "PropulsionModel", "FlightState"]
