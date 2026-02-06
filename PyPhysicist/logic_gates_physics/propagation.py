"""Propagation delay models using distributed RC and telegrapher equations."""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math


@dataclass(frozen=True)
class DistributedRCParameters:
    """Per-unit-length interconnect parameters."""

    resistance_per_m: float
    capacitance_per_m: float
    inductance_per_m: float | None = None
    conductance_per_m: float | None = None


@dataclass(frozen=True)
class PropagationDelayResult:
    """Computed delay and related metrics for an interconnect."""

    length_m: float
    delay_s: float
    characteristic_impedance_ohm: float | None
    propagation_velocity_m_s: float | None
    attenuation_nepers_m: float | None


def rc_elmore_delay(length_m: float, params: DistributedRCParameters) -> PropagationDelayResult:
    """Estimate delay using distributed RC Elmore approximation.

    This uses R' and C' (per-unit-length) to estimate the 50% delay for a
    step input as ~0.38 * R' * C' * L^2.
    """

    if length_m <= 0:
        raise ValueError("Length must be positive.")
    if params.resistance_per_m <= 0 or params.capacitance_per_m <= 0:
        raise ValueError("RC parameters must be positive.")

    delay = 0.38 * params.resistance_per_m * params.capacitance_per_m * length_m**2
    return PropagationDelayResult(
        length_m=length_m,
        delay_s=delay,
        characteristic_impedance_ohm=None,
        propagation_velocity_m_s=None,
        attenuation_nepers_m=None,
    )


def telegrapher_propagation(
    length_m: float,
    params: DistributedRCParameters,
    frequency_hz: float,
) -> PropagationDelayResult:
    """Approximate delay and attenuation from telegrapher equations.

    Uses complex propagation constant gamma = sqrt((R + j w L)(G + j w C)).
    """

    if length_m <= 0:
        raise ValueError("Length must be positive.")
    if frequency_hz <= 0:
        raise ValueError("Frequency must be positive.")
    if params.resistance_per_m <= 0 or params.capacitance_per_m <= 0:
        raise ValueError("RC parameters must be positive.")

    r = params.resistance_per_m
    c = params.capacitance_per_m
    l = params.inductance_per_m or 0.0
    g = params.conductance_per_m or 0.0

    omega = 2.0 * math.pi * frequency_hz
    a = complex(r, omega * l)
    b = complex(g, omega * c)
    gamma = cmath.sqrt(a * b)

    alpha = gamma.real
    beta = gamma.imag if gamma.imag != 0 else omega * math.sqrt(l * c) if l and c else None
    if beta:
        velocity = omega / beta
    else:
        velocity = None

    delay = length_m / velocity if velocity else None
    impedance = cmath.sqrt(a / b).real if l or g else (math.sqrt(l / c) if l and c else None)

    return PropagationDelayResult(
        length_m=length_m,
        delay_s=delay if delay else float("nan"),
        characteristic_impedance_ohm=impedance,
        propagation_velocity_m_s=velocity,
        attenuation_nepers_m=alpha,
    )
