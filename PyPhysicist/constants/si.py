"""SI physical constants and reference values.

All constants are expressed in SI units. Use these values to ensure consistent
units when applying formulas. For dimensional safety in complex workflows,
consider wrapping values in :class:`PyPhysicist.units.Quantity` or using Pint.
"""

AVOGADRO_CONSTANT = 6.02214076e23  # mol^-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
COULOMB_CONSTANT = 8.9875517923e9  # N·m^2/C^2
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3·kg^-1·s^-2
IDEAL_GAS_CONSTANT = 8.314462618  # J/(mol·K)
NEUTRON_MASS = 1.67492749804e-27  # kg
PLANCK_CONSTANT = 6.62607015e-34  # J·s
PLANCK_REDUCED_CONSTANT = 1.054571817e-34  # J·s
PLANCK_TIME = 5.39124e-44  # s
PLANCK_LENGTH = 1.616255e-35  # m
PLANCK_TEMPERATURE = 1.416784e32  # K
PLANCK_MASS = 2.176434e-8  # kg
PROTON_MASS = 1.67262192369e-27  # kg
SPEED_OF_LIGHT = 299_792_458  # m/s
STANDARD_GRAVITY = 9.80665  # m/s^2

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
]
