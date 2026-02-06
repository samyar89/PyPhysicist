"""Idealized staging physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from PyPhysicist.units import Quantity
from PyPhysicist.units.conversion import coerce_value, wrap_quantity


@dataclass(frozen=True)
class Stage:
    """Mass properties for an idealized stage."""

    dry_mass: Quantity | float
    propellant_mass: Quantity | float

    def total_mass(self) -> Quantity | float:
        dry_value, _ = coerce_value(self.dry_mass, "kg", name="dry_mass")
        prop_value, _ = coerce_value(self.propellant_mass, "kg", name="propellant_mass")
        total = dry_value + prop_value
        return wrap_quantity(total, "kg", self.dry_mass, self.propellant_mass)


def multi_stage_mass(stages: Iterable[Stage], payload_mass: Quantity | float = 0.0) -> List[Quantity | float]:
    """Return cumulative masses for each stage including payload."""

    payload_value, _ = coerce_value(payload_mass, "kg", name="payload_mass")
    totals = []
    running = payload_value
    for stage in reversed(list(stages)):
        running = running + coerce_value(stage.total_mass(), "kg", name="stage_total")[0]
        totals.append(running)
    totals = list(reversed(totals))
    return [wrap_quantity(value, "kg", payload_mass) for value in totals]


def instantaneous_staging_mass(initial_mass: Quantity | float, dry_mass_drop: Quantity | float) -> Quantity | float:
    """Compute mass immediately after staging drop."""

    m0_value, _ = coerce_value(initial_mass, "kg", name="initial_mass")
    drop_value, _ = coerce_value(dry_mass_drop, "kg", name="dry_mass_drop")
    remaining = m0_value - drop_value
    return wrap_quantity(remaining, "kg", initial_mass, dry_mass_drop)


def delta_v_partitioning(mass_ratios: Iterable[float], exhaust_velocities: Iterable[Quantity | float]) -> List[Quantity | float]:
    """Partition delta-v across stages using Tsiolkovsky form."""

    delta_vs = []
    for ratio, ve in zip(mass_ratios, exhaust_velocities):
        ve_value, _ = coerce_value(ve, "m/s", name="exhaust_velocity")
        delta_v = ve_value * np.log(ratio)
        delta_vs.append(wrap_quantity(delta_v, "m/s", ve))
    return delta_vs


def optimal_mass_ratio(total_mass_ratio: float, stages: int) -> float:
    """Optimal equal mass ratio per stage for identical exhaust velocity."""

    if stages <= 0:
        raise ValueError("stages must be positive")
    return total_mass_ratio ** (1.0 / stages)


__all__ = [
    "Stage",
    "multi_stage_mass",
    "instantaneous_staging_mass",
    "delta_v_partitioning",
    "optimal_mass_ratio",
]
