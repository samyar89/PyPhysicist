"""PyPhysicist public API."""

from . import constants
from .constants import (
    AVOGADRO_CONSTANT,
    BOLTZMANN_CONSTANT,
    COULOMB_CONSTANT,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS,
    GRAVITATIONAL_CONSTANT,
    IDEAL_GAS_CONSTANT,
    NEUTRON_MASS,
    PLANCK_CONSTANT,
    PLANCK_LENGTH,
    PLANCK_MASS,
    PLANCK_REDUCED_CONSTANT,
    PLANCK_TEMPERATURE,
    PLANCK_TIME,
    PROTON_MASS,
    SPEED_OF_LIGHT,
    STANDARD_GRAVITY,
)
from .electromagnetism.circuits import (
    I,
    R,
    V,
    current,
    resistance,
    resistance_parallel,
    resistance_series,
    voltage,
)
from .electromagnetism.electrostatics import capacitance, coulomb_force, electric_field
from .mechanics.dynamics import (
    F,
    Weight,
    force,
    momentum,
    newton_second_law,
    weight,
    centripetal_force,
)
from .mechanics.energy import (
    ELASTIC_POTENTIAL_ENERGY,
    GRAVITATIONAL_POTENTIAL_ENERGY,
    KINETIC_ENERGY,
    MECHANICAL_ENERGY,
    Work,
    elastic_potential_energy,
    gravitational_potential_energy,
    kinetic_energy,
    mechanical_energy,
    spring_potential_energy,
    work,
)
from .mechanics.kinematics import (
    Velocity,
    centripetal_acceleration,
    velocity,
)
from .latex import DEFAULT_FORMULAS, LatexCalculation, LatexConverter, LatexFormula
from .geometry import (
    as_tensor,
    christoffel_symbols,
    covariant_derivative_vector,
    lower_index,
    metric_derivatives,
    metric_inverse,
    partial_derivative,
    raise_index,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
)
from .relativity.gravity import Schwarzschild_radius, schwarzschild_radius
from .relativity.special import (
    length_contraction,
    relativistic_energy,
    time_dilation,
)
from .thermodynamics.heat import entropy_change, heat_capacity
from .thermodynamics.ideal_gases import ideal_gas_pressure
from .units import Quantity, UnitError
from .waves_optics.optics import refractive_index
from .waves_optics.waves import frequency, wave_power, wavelength
from . import optimal_control
from . import quantum_chaos
from . import combinatorial_dynamics
from . import astro_numerics
from . import logic_gates_physics
from . import physlearn
from . import fluids
<<<<<<< codex/add-classical-astrodynamics-submodule-to-pyphysicist
from . import astrodynamics
=======
from . import rocketry
>>>>>>> main

__all__ = [
    "AVOGADRO_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "COULOMB_CONSTANT",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "GRAVITATIONAL_CONSTANT",
    "IDEAL_GAS_CONSTANT",
    "NEUTRON_MASS",
    "PLANCK_CONSTANT",
    "PLANCK_LENGTH",
    "PLANCK_MASS",
    "PLANCK_REDUCED_CONSTANT",
    "PLANCK_TEMPERATURE",
    "PLANCK_TIME",
    "PROTON_MASS",
    "SPEED_OF_LIGHT",
    "STANDARD_GRAVITY",
    "Quantity",
    "UnitError",
    "capacitance",
    "centripetal_acceleration",
    "centripetal_force",
    "coulomb_force",
    "current",
    "electric_field",
    "elastic_potential_energy",
    "entropy_change",
    "force",
    "frequency",
    "gravitational_potential_energy",
    "heat_capacity",
    "ideal_gas_pressure",
    "kinetic_energy",
    "length_contraction",
    "mechanical_energy",
    "momentum",
    "newton_second_law",
    "refractive_index",
    "relativistic_energy",
    "resistance",
    "resistance_parallel",
    "resistance_series",
    "schwarzschild_radius",
    "spring_potential_energy",
    "time_dilation",
    "velocity",
    "voltage",
    "wave_power",
    "wavelength",
    "weight",
    "work",
    "constants",
    "optimal_control",
    "quantum_chaos",
    "combinatorial_dynamics",
    "astro_numerics",
    "logic_gates_physics",
    "physlearn",
    "fluids",
<<<<<<< codex/add-classical-astrodynamics-submodule-to-pyphysicist
    "astrodynamics",
=======
    "rocketry",
>>>>>>> main
    "V",
    "I",
    "R",
    "F",
    "Velocity",
    "Weight",
    "Work",
    "Schwarzschild_radius",
    "KINETIC_ENERGY",
    "GRAVITATIONAL_POTENTIAL_ENERGY",
    "MECHANICAL_ENERGY",
    "ELASTIC_POTENTIAL_ENERGY",
    "LatexCalculation",
    "LatexConverter",
    "LatexFormula",
    "DEFAULT_FORMULAS",
    "as_tensor",
    "metric_inverse",
    "lower_index",
    "raise_index",
    "partial_derivative",
    "metric_derivatives",
    "christoffel_symbols",
    "covariant_derivative_vector",
    "riemann_tensor",
    "ricci_tensor",
    "scalar_curvature",
]
