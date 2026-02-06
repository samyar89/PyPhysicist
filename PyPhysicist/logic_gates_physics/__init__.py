"""Logic gate physics models for computational limits."""

from .thermodynamics import (
    GateDissipationAnalysis,
    LandauerComparison,
    landauer_limit,
    analyze_gate_dissipation,
)
from .propagation import (
    DistributedRCParameters,
    PropagationDelayResult,
    rc_elmore_delay,
    telegrapher_propagation,
)
from .quantum_effects import (
    QuantumLeakageResult,
    QuantumCoherenceResult,
    tunneling_leakage_current,
    coherence_time_limit,
)

__all__ = [
    "GateDissipationAnalysis",
    "LandauerComparison",
    "landauer_limit",
    "analyze_gate_dissipation",
    "DistributedRCParameters",
    "PropagationDelayResult",
    "rc_elmore_delay",
    "telegrapher_propagation",
    "QuantumLeakageResult",
    "QuantumCoherenceResult",
    "tunneling_leakage_current",
    "coherence_time_limit",
]
