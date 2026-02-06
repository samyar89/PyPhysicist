"""Classical rocket equations in operator form."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity

from .core import FlightState, PropulsionModel, RocketBody
from .environment import DragModel, GravityModel


@dataclass(frozen=True)
class ResidualOperator:
    """Residual operator for analytic inspection and evaluation."""

    description: str
    formula: str
    evaluate: Callable[..., Any]
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def tsiolkovsky_equation(
    rocket: RocketBody,
    propulsion: PropulsionModel,
    *,
    delta_v: Quantity | float | None = None,
    initial_mass: Quantity | float | None = None,
    final_mass: Quantity | float | None = None,
) -> ResidualOperator:
    """Return the Tsiolkovsky rocket equation residual operator."""

    assumptions = (
        "constant exhaust velocity",
        "no external forces",
        "point mass",
    )

    def _evaluate() -> Any:
        m0 = initial_mass if initial_mass is not None else rocket.total_mass()
        mf = final_mass if final_mass is not None else rocket.dry_mass
        dv = delta_v if delta_v is not None else 0.0
        m0_value, _ = coerce_value(m0, "kg", name="initial_mass")
        mf_value, _ = coerce_value(mf, "kg", name="final_mass")
        dv_value, _ = coerce_value(dv, "m/s", name="delta_v")
        ve_value, _ = coerce_value(
            propulsion.effective_exhaust_velocity(), "m/s", name="exhaust_velocity"
        )
        residual = dv_value - ve_value * np.log(m0_value / mf_value)
        return wrap_quantity(residual, "m/s", dv, m0, mf, propulsion.exhaust_velocity)

    formula = "Δv - v_e ln(m0/mf) = 0"
    return ResidualOperator("Tsiolkovsky rocket equation", formula, _evaluate, assumptions)


def meshchersky_equation(
    rocket: RocketBody,
    propulsion: PropulsionModel,
    *,
    state: FlightState,
    external_force: Any = 0.0,
) -> ResidualOperator:
    """Variable-mass equation (Meshchersky form) residual."""

    assumptions = (
        "variable mass",
        "exhaust momentum aligned with velocity",
    )

    def _evaluate() -> Any:
        mass_value, _ = coerce_value(state.mass, "kg", name="mass")
        accel_value, _ = coerce_value(state.acceleration, "m/s^2", name="acceleration")
        mdot = rocket.mass_rate(state.time, state)
        mdot_value, _ = coerce_value(mdot, "kg/s", name="mass_flow_rate")
        ve_value, _ = coerce_value(
            propulsion.effective_exhaust_velocity(), "m/s", name="exhaust_velocity"
        )
        external_value, _ = coerce_value(external_force, "N", name="external_force")
        residual = mass_value * accel_value - external_value - ve_value * mdot_value
        return wrap_quantity(residual, "N", state.mass, state.acceleration, mdot, external_force)

    formula = "m a - F_ext - v_e \dot{m} = 0"
    return ResidualOperator("Meshchersky equation", formula, _evaluate, assumptions)


def thrust_drag_gravity_equation(
    rocket: RocketBody,
    propulsion: PropulsionModel,
    *,
    state: FlightState,
    gravity: GravityModel,
    drag: DragModel | None = None,
    atmosphere_density: Any = 0.0,
) -> ResidualOperator:
    """Equation of motion including thrust, drag, and gravity."""

    assumptions = (
        "point mass",
        "forces resolved in inertial frame",
    )

    def _evaluate() -> Any:
        mass_value, _ = coerce_value(state.mass, "kg", name="mass")
        accel_value, _ = coerce_value(state.acceleration, "m/s^2", name="acceleration")
        thrust = propulsion.thrust(state.time, state, mass_flow_rate=rocket.mass_rate(state.time, state))
        thrust_value, _ = coerce_value(thrust, "N", name="thrust")
        gravity_accel = gravity.acceleration(state.position)
        gravity_value, _ = coerce_value(gravity_accel, "m/s^2", name="gravity")
        force = thrust_value + mass_value * gravity_value
        if drag is not None:
            drag_force = drag.force(atmosphere_density, state.velocity, rocket.reference_area)
            drag_value, _ = coerce_value(drag_force, "N", name="drag")
            force = force + drag_value
        residual = mass_value * accel_value - force
        return wrap_quantity(residual, "N", state.mass, state.acceleration, thrust, gravity_accel)

    formula = "m a - (T + D + m g) = 0"
    return ResidualOperator("Thrust-drag-gravity equation", formula, _evaluate, assumptions)


def vertical_ascent_equation(
    rocket: RocketBody,
    propulsion: PropulsionModel,
    *,
    state: FlightState,
    gravity: GravityModel,
    drag: DragModel | None = None,
    atmosphere_density: Any = 0.0,
) -> ResidualOperator:
    """Vertical ascent equation residual."""

    assumptions = (
        "one-dimensional motion",
        "thrust aligned with vertical axis",
    )

    def _evaluate() -> Any:
        mass_value, _ = coerce_value(state.mass, "kg", name="mass")
        accel_value, _ = coerce_value(state.acceleration, "m/s^2", name="acceleration")
        thrust = propulsion.thrust(state.time, state, mass_flow_rate=rocket.mass_rate(state.time, state))
        thrust_value, _ = coerce_value(thrust, "N", name="thrust")
        gravity_accel = gravity.acceleration(state.position)
        gravity_value, _ = coerce_value(gravity_accel, "m/s^2", name="gravity")
        force = thrust_value + mass_value * gravity_value
        if drag is not None:
            drag_force = drag.force(atmosphere_density, state.velocity, rocket.reference_area)
            drag_value, _ = coerce_value(drag_force, "N", name="drag")
            force = force + drag_value
        residual = mass_value * accel_value - force
        return wrap_quantity(residual, "N", state.mass, state.acceleration, thrust, gravity_accel)

    formula = "m dv/dt - (T - D - m g) = 0"
    return ResidualOperator("Vertical ascent equation", formula, _evaluate, assumptions)


def planar_flight_equations(
    rocket: RocketBody,
    propulsion: PropulsionModel,
    *,
    state: FlightState,
    gravity: GravityModel,
    drag: DragModel | None = None,
    atmosphere_density: Any = 0.0,
) -> ResidualOperator:
    """Planar flight equation residual (2D)."""

    assumptions = (
        "planar motion",
        "forces resolved in plane",
    )

    def _evaluate() -> Any:
        mass_value, _ = coerce_value(state.mass, "kg", name="mass")
        accel_value, _ = coerce_value(state.acceleration, "m/s^2", name="acceleration")
        thrust = propulsion.thrust(state.time, state, mass_flow_rate=rocket.mass_rate(state.time, state))
        thrust_value, _ = coerce_value(thrust, "N", name="thrust")
        gravity_accel = gravity.acceleration(state.position)
        gravity_value, _ = coerce_value(gravity_accel, "m/s^2", name="gravity")
        force = thrust_value + mass_value * gravity_value
        if drag is not None:
            drag_force = drag.force(atmosphere_density, state.velocity, rocket.reference_area)
            drag_value, _ = coerce_value(drag_force, "N", name="drag")
            force = force + drag_value
        residual = mass_value * accel_value - force
        return wrap_quantity(residual, "N", state.mass, state.acceleration, thrust, gravity_accel)

    formula = "m a - (T + D + m g) = 0"
    return ResidualOperator("Planar flight equations", formula, _evaluate, assumptions)


__all__ = [
    "ResidualOperator",
    "tsiolkovsky_equation",
    "meshchersky_equation",
    "thrust_drag_gravity_equation",
    "vertical_ascent_equation",
    "planar_flight_equations",
]
