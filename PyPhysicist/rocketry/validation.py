"""Diagnostics and validation checks for rocket models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from PyPhysicist.units.conversion import coerce_value, wrap_quantity

from .core import FlightState, RocketBody
from .staging import Stage


@dataclass(frozen=True)
class ValidationEntry:
    name: str
    passed: bool
    details: str
    value: Any | None = None


@dataclass(frozen=True)
class ValidationReport:
    entries: List[ValidationEntry] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(entry.passed for entry in self.entries)


def validate_mass_decrease(rocket: RocketBody, *, t0: float, t1: float, state: FlightState | None = None) -> ValidationReport:
    """Verify mass decreases monotonically between two times."""

    m0 = rocket.mass(t0, t0=t0, state=state)
    m1 = rocket.mass(t1, t0=t0, state=state)
    m0_value, _ = coerce_value(m0, "kg", name="mass_t0")
    m1_value, _ = coerce_value(m1, "kg", name="mass_t1")
    passed = m1_value <= m0_value
    entry = ValidationEntry(
        "mass decrease",
        passed,
        "mass should decrease or remain constant as propellant is expended",
        wrap_quantity(m1_value - m0_value, "kg", m0, m1),
    )
    return ValidationReport([entry])


def validate_stage_mass_conservation(stages: List[Stage], payload_mass: Any = 0.0) -> ValidationReport:
    """Ensure total mass equals sum of stage and payload masses."""

    payload_value, _ = coerce_value(payload_mass, "kg", name="payload_mass")
    total = payload_value
    for stage in stages:
        total = total + coerce_value(stage.total_mass(), "kg", name="stage_total")[0]
    components = payload_value + sum(
        coerce_value(stage.dry_mass, "kg", name="dry_mass")[0]
        + coerce_value(stage.propellant_mass, "kg", name="propellant_mass")[0]
        for stage in stages
    )
    passed = abs(total - components) <= 1e-9
    entry = ValidationEntry(
        "stage mass conservation",
        passed,
        "total mass equals payload plus stage dry and propellant masses",
        wrap_quantity(total, "kg", payload_mass),
    )
    return ValidationReport([entry])


def validate_state_consistency(state: FlightState) -> ValidationReport:
    """Check state values for physical admissibility."""

    mass_value, _ = coerce_value(state.mass, "kg", name="mass")
    time_value, _ = coerce_value(state.time, "s", name="time")
    entries = []
    entries.append(
        ValidationEntry(
            "positive mass",
            mass_value > 0,
            "mass must remain positive",
            state.mass,
        )
    )
    entries.append(
        ValidationEntry(
            "non-negative time",
            time_value >= 0,
            "time should be non-negative",
            state.time,
        )
    )
    return ValidationReport(entries)


__all__ = [
    "ValidationEntry",
    "ValidationReport",
    "validate_mass_decrease",
    "validate_stage_mass_conservation",
    "validate_state_consistency",
]
