# PyPhysicist

A lightweight collection of physics formulas implemented in Python, covering
mechanics, electromagnetism, thermodynamics, waves/optics, relativity, and common
energy calculations. The library now ships with a domain-oriented package layout,
SI constants, and a lightweight quantity helper for dimensional awareness.

## نصب (Installation)

```bash
pip install PyPhysicist
```

> **Note**: When working from source, ensure the repository root is on your
> `PYTHONPATH`, or install with `pip install -e .`.

## Quick start

```python
import PyPhysicist as pp

# Mechanics
force = pp.newton_second_law(mass=2.0, acceleration=9.81)

# Electromagnetism
voltage = pp.voltage(current=0.5, resistance=10.0)

# Relativity
energy = pp.relativistic_energy(mass=1.0)

print(force, voltage, energy)
```

## Example calculations

```python
from PyPhysicist import momentum, ideal_gas_pressure, refractive_index

p = momentum(mass=3.2, velocity=4.5)
pressure = ideal_gas_pressure(n=1.0, t=300.0, v=0.024)
index = refractive_index(speed_of_light=299_792_458, medium_speed=200_000_000)

print(p, pressure, index)
```

## Documentation

Build the documentation locally with Sphinx:

```bash
pip install sphinx
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in your browser.

## Development

Run the test suite with pytest:

```bash
pip install pytest
pytest
```

## ✅ Technical recommendations (library core)

1. Clean, extensible API design

   The library now ships a domain-first layout (for example: `PyPhysicist/mechanics/kinematics.py`, `PyPhysicist/electromagnetism/circuits.py`, `PyPhysicist/constants/si.py`) to keep growth organized.

   PEP8 naming is the default (snake_case functions like `schwarzschild_radius` or `kinetic_energy`). Symbolic aliases like `V`, `I`, `R`, or `KINETIC_ENERGY` remain available for discoverability.

   `__all__` and targeted re-exports now clarify the public API surface.

2. Units management and dimensional-safety

   A lightweight `Quantity` type is included under `PyPhysicist.units`. Formula helpers now accept `Quantity` inputs, validate dimensional compatibility, and return `Quantity` outputs when unit-tagged values are provided.

   The bundled converter (`convert_value`) handles common SI unit conversions and raises `UnitError` on incompatible units.

3. NumPy and array support

   Ensure all functions (such as `KINETIC_ENERGY` or `Velocity`) are written to work with NumPy arrays as well (simple arithmetic often already works, but it is better to guarantee and document this).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
