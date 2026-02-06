"""Classical rocket physics submodule."""

from .core import FlightState, PropulsionModel, RocketBody
from .dimensionless import (
    ballistic_coefficient,
    characteristic_velocity_ratio,
    mass_ratio,
    thrust_to_weight,
)
from .environment import (
    AtmosphereModel,
    DragModel,
    ExponentialAtmosphere,
    GravityModel,
    InverseSquareGravity,
    LayeredAtmosphere,
    LiftModel,
    UniformGravity,
)
from .equations import (
    ResidualOperator,
    meshchersky_equation,
    planar_flight_equations,
    thrust_drag_gravity_equation,
    tsiolkovsky_equation,
    vertical_ascent_equation,
)
from .staging import (
    Stage,
    delta_v_partitioning,
    instantaneous_staging_mass,
    multi_stage_mass,
    optimal_mass_ratio,
)
from .theorems import (
    TheoremResult,
    energy_ascent_bound,
    gravity_drag_loss,
    ideal_rocket_efficiency,
    vacuum_vs_atmospheric_performance,
)
from .trajectories import (
    TrajectorySolution,
    constant_thrust_planar_range,
    orbital_insertion_condition,
    vacuum_ascent_altitude,
    vertical_ascent_velocity,
)
from .validation import (
    ValidationEntry,
    ValidationReport,
    validate_mass_decrease,
    validate_stage_mass_conservation,
    validate_state_consistency,
)

__all__ = [
    "FlightState",
    "PropulsionModel",
    "RocketBody",
    "ballistic_coefficient",
    "characteristic_velocity_ratio",
    "mass_ratio",
    "thrust_to_weight",
    "AtmosphereModel",
    "DragModel",
    "ExponentialAtmosphere",
    "GravityModel",
    "InverseSquareGravity",
    "LayeredAtmosphere",
    "LiftModel",
    "UniformGravity",
    "ResidualOperator",
    "meshchersky_equation",
    "planar_flight_equations",
    "thrust_drag_gravity_equation",
    "tsiolkovsky_equation",
    "vertical_ascent_equation",
    "Stage",
    "delta_v_partitioning",
    "instantaneous_staging_mass",
    "multi_stage_mass",
    "optimal_mass_ratio",
    "TheoremResult",
    "energy_ascent_bound",
    "gravity_drag_loss",
    "ideal_rocket_efficiency",
    "vacuum_vs_atmospheric_performance",
    "TrajectorySolution",
    "constant_thrust_planar_range",
    "orbital_insertion_condition",
    "vacuum_ascent_altitude",
    "vertical_ascent_velocity",
    "ValidationEntry",
    "ValidationReport",
    "validate_mass_decrease",
    "validate_stage_mass_conservation",
    "validate_state_consistency",
]
