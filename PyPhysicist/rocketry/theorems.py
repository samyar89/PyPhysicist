"""Classical limits and theorems for rocket analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


@dataclass(frozen=True)
class TheoremResult:
    """Result of an analytical theorem evaluation."""

    description: str
    formula: str
    value: Any
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def ideal_rocket_efficiency(delta_v: Quantity | float, exhaust_velocity: Quantity | float) -> TheoremResult:
    """Ideal propulsive efficiency for a rocket with constant exhaust velocity."""

    dv_value, _ = coerce_value(delta_v, "m/s", name="delta_v")
    ve_value, _ = coerce_value(exhaust_velocity, "m/s", name="exhaust_velocity")
    ratio = dv_value / ve_value
    efficiency = ratio * np.exp(-ratio)
    result = wrap_quantity(efficiency, "1", delta_v, exhaust_velocity)
    return TheoremResult(
        "ideal rocket efficiency limit",
        "η = (Δv/ve) exp(-Δv/ve)",
        result,
        ("constant exhaust velocity", "no external forces"),
    )


def gravity_drag_loss(delta_v_actual: Quantity | float, delta_v_ideal: Quantity | float) -> TheoremResult:
    """Compute combined gravity and drag losses as a delta-v gap."""

    actual_value, _ = coerce_value(delta_v_actual, "m/s", name="delta_v_actual")
    ideal_value, _ = coerce_value(delta_v_ideal, "m/s", name="delta_v_ideal")
    loss = actual_value - ideal_value
    result = wrap_quantity(loss, "m/s", delta_v_actual, delta_v_ideal)
    return TheoremResult(
        "gravity and drag loss",
        "Δv_loss = Δv_actual - Δv_ideal",
        result,
        ("losses lumped",),
    )


def energy_ascent_bound(mu: Quantity | float, radius: Quantity | float, initial_speed: Quantity | float = 0.0) -> TheoremResult:
    """Energy-based lower bound on required velocity for escape."""

    mu_value, _ = coerce_value(mu, "m^3/s^2", name="mu")
    r_value, _ = coerce_value(radius, "m", name="radius")
    v0_value, _ = coerce_value(initial_speed, "m/s", name="initial_speed")
    escape = np.sqrt(2 * mu_value / r_value)
    required = escape - v0_value
    result = wrap_quantity(required, "m/s", mu, radius, initial_speed)
    return TheoremResult(
        "energy ascent bound",
        "v_req = sqrt(2 μ/r) - v0",
        result,
        ("two-body gravity", "no drag"),
    )


def vacuum_vs_atmospheric_performance(delta_v_vacuum: Quantity | float, delta_v_atmosphere: Quantity | float) -> TheoremResult:
    """Compare vacuum and atmospheric delta-v performance."""

    vac_value, _ = coerce_value(delta_v_vacuum, "m/s", name="delta_v_vacuum")
    atm_value, _ = coerce_value(delta_v_atmosphere, "m/s", name="delta_v_atmosphere")
    penalty = vac_value - atm_value
    result = wrap_quantity(penalty, "m/s", delta_v_vacuum, delta_v_atmosphere)
    return TheoremResult(
        "vacuum vs atmospheric performance",
        "Δv_penalty = Δv_vacuum - Δv_atmosphere",
        result,
        ("same mass ratio",),
    )


__all__ = [
    "TheoremResult",
    "ideal_rocket_efficiency",
    "gravity_drag_loss",
    "energy_ascent_bound",
    "vacuum_vs_atmospheric_performance",
]
