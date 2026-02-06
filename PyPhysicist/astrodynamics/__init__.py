"""Classical astrodynamics submodule."""

from .core import CelestialBody, Orbit, OrbitState
from .dimensionless import (
    angular_momentum_ratio,
    energy_ratio,
    escape_parameter,
    normalized_time_scale,
)
from .elements import OrbitalElements, elements_to_state, state_to_elements
from .equations import (
    central_force_equation,
    polar_component_equations,
    reduced_mass_equation,
    two_body_equation,
)
from .invariants import (
    angular_momentum_vector,
    areal_velocity,
    laplace_runge_lenz_vector,
    specific_orbital_energy,
)
from .limits import (
    bound_unbound_comparison,
    nearly_circular_velocity,
    scaling_arguments,
    small_eccentricity_radius,
)
from .solutions import (
    circular_orbit,
    elliptic_kepler_orbit,
    hyperbolic_flyby,
    parabolic_escape,
)
from .theorems import (
    energy_condition,
    escape_velocity,
    kepler_first_law,
    kepler_second_law,
    kepler_third_law,
    vis_viva,
    virial_theorem,
)
from .validation import (
    ValidationReport,
    validate_conservation,
    validate_keplerian_constraints,
    validate_regime,
)

__all__ = [
    "CelestialBody",
    "OrbitState",
    "Orbit",
    "OrbitalElements",
    "elements_to_state",
    "state_to_elements",
    "two_body_equation",
    "reduced_mass_equation",
    "central_force_equation",
    "polar_component_equations",
    "specific_orbital_energy",
    "angular_momentum_vector",
    "laplace_runge_lenz_vector",
    "areal_velocity",
    "circular_orbit",
    "elliptic_kepler_orbit",
    "parabolic_escape",
    "hyperbolic_flyby",
    "kepler_first_law",
    "kepler_second_law",
    "kepler_third_law",
    "virial_theorem",
    "escape_velocity",
    "vis_viva",
    "energy_condition",
    "small_eccentricity_radius",
    "nearly_circular_velocity",
    "bound_unbound_comparison",
    "scaling_arguments",
    "energy_ratio",
    "angular_momentum_ratio",
    "escape_parameter",
    "normalized_time_scale",
    "ValidationReport",
    "validate_conservation",
    "validate_keplerian_constraints",
    "validate_regime",
]
