"""Quantum effects for ultra-scaled and reversible logic gates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PyPhysicist.constants import PLANCK_REDUCED_CONSTANT


@dataclass(frozen=True)
class QuantumLeakageResult:
    """Result for tunneling leakage estimation."""

    barrier_height_ev: float
    oxide_thickness_m: float
    electric_field_v_m: float
    leakage_current_density_a_m2: float
    transmission_probability: float


@dataclass(frozen=True)
class QuantumCoherenceResult:
    """Result for coherence time estimation."""

    t1_s: float
    t2_s: float
    gate_time_s: float
    coherence_margin: float


def tunneling_leakage_current(
    barrier_height_ev: float,
    oxide_thickness_m: float,
    electric_field_v_m: float,
    effective_mass_ratio: float = 0.5,
) -> QuantumLeakageResult:
    """Estimate leakage via a WKB tunneling approximation.

    This simplified model uses a Fowler-Nordheim style expression for
    transmission probability and current density.
    """

    if barrier_height_ev <= 0:
        raise ValueError("Barrier height must be positive.")
    if oxide_thickness_m <= 0:
        raise ValueError("Oxide thickness must be positive.")
    if electric_field_v_m <= 0:
        raise ValueError("Electric field must be positive.")
    if effective_mass_ratio <= 0:
        raise ValueError("Effective mass ratio must be positive.")

    electron_mass = 9.1093837015e-31
    q = 1.602176634e-19
    barrier_j = barrier_height_ev * q
    effective_mass = electron_mass * effective_mass_ratio

    exponent = -2.0 * oxide_thickness_m * math.sqrt(2.0 * effective_mass * barrier_j) / PLANCK_REDUCED_CONSTANT
    transmission = math.exp(exponent)

    # Current density scaling (simplified Fowler-Nordheim style).
    current_density = (q**3 * electric_field_v_m**2) / (16.0 * math.pi**2 * PLANCK_REDUCED_CONSTANT * barrier_j)
    current_density *= transmission

    return QuantumLeakageResult(
        barrier_height_ev=barrier_height_ev,
        oxide_thickness_m=oxide_thickness_m,
        electric_field_v_m=electric_field_v_m,
        leakage_current_density_a_m2=current_density,
        transmission_probability=transmission,
    )


def coherence_time_limit(t1_s: float, t2_s: float, gate_time_s: float) -> QuantumCoherenceResult:
    """Compute coherence margin for reversible or quantum gates.

    The margin > 1 implies the gate fits within coherence time.
    """

    if t1_s <= 0 or t2_s <= 0 or gate_time_s <= 0:
        raise ValueError("Times must be positive.")

    effective_coherence = min(t1_s, t2_s)
    margin = effective_coherence / gate_time_s

    return QuantumCoherenceResult(
        t1_s=t1_s,
        t2_s=t2_s,
        gate_time_s=gate_time_s,
        coherence_margin=margin,
    )
