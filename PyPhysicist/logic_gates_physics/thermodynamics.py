"""Thermodynamic analysis for gate switching energy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PyPhysicist.constants import BOLTZMANN_CONSTANT


@dataclass(frozen=True)
class LandauerComparison:
    """Comparison to the Landauer limit for a single bit operation."""

    temperature: float
    landauer_limit_joule: float
    energy_per_op_joule: float
    ratio_to_landauer: float
    margin_joule: float


@dataclass(frozen=True)
class GateDissipationAnalysis:
    """Summary of thermodynamic energy dissipation analysis."""

    comparison: LandauerComparison
    activity_factor: float
    operations_per_second: float
    average_power_watt: float


def landauer_limit(temperature: float) -> float:
    """Return the Landauer limit (k_B T ln 2) in joules."""

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return BOLTZMANN_CONSTANT * temperature * math.log(2.0)


def analyze_gate_dissipation(
    temperature: float,
    energy_per_op_joule: float,
    operations_per_second: float,
    activity_factor: float = 1.0,
) -> GateDissipationAnalysis:
    """Analyze gate dissipation against the Landauer limit.

    Args:
        temperature: Temperature in kelvin.
        energy_per_op_joule: Energy dissipated per logical operation.
        operations_per_second: Switching rate in operations per second.
        activity_factor: Fraction of gates switching per cycle.
    """

    if energy_per_op_joule <= 0:
        raise ValueError("Energy per operation must be positive.")
    if operations_per_second <= 0:
        raise ValueError("Operations per second must be positive.")
    if not 0 < activity_factor <= 1.0:
        raise ValueError("Activity factor must be between 0 and 1.")

    limit = landauer_limit(temperature)
    ratio = energy_per_op_joule / limit
    comparison = LandauerComparison(
        temperature=temperature,
        landauer_limit_joule=limit,
        energy_per_op_joule=energy_per_op_joule,
        ratio_to_landauer=ratio,
        margin_joule=energy_per_op_joule - limit,
    )
    average_power = energy_per_op_joule * operations_per_second * activity_factor
    return GateDissipationAnalysis(
        comparison=comparison,
        activity_factor=activity_factor,
        operations_per_second=operations_per_second,
        average_power_watt=average_power,
    )
