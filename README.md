# PyPhysicist

A lightweight collection of physics formulas implemented in Python, covering
mechanics, electromagnetism, thermodynamics, waves/optics, relativity, and common
energy calculations.

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
voltage = pp.V(I=0.5, R=10.0)

# Relativity
energy = pp.relativistic_energy(mass=1.0, c=299_792_458)

print(force, voltage, energy)
```

## Example calculations

```python
from PyPhysicist import (
    momentum,
    ideal_gas_pressure,
    refractive_index,
)

p = momentum(mass=3.2, velocity=4.5)
pressure = ideal_gas_pressure(n=1.0, r=8.314, t=300.0, v=0.024)
index = refractive_index(c=299_792_458, v=200_000_000)

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

   Split modules into subpackages (for example: `mechanics/`, `electricity/`, `relativity/`, `constants/`) so the structure stays clean as the project grows. Right now everything is in `__init__.py`, which makes extension harder.

   Use PEP8 naming (snake_case functions like `schwarzschild_radius` or `kinetic_energy`) to align with the Python ecosystem; currently there are functions such as `V`, `I`, `R`, `KINETIC_ENERGY` with mixed styles.

   Add `__all__` and targeted re-exports so consumers can see what is public (especially once modules are split).

2. Units management and dimensional-safety

   Add unit support (with `pint` or `astropy.units`) so users cannot accidentally provide values in the wrong units; this is critical for a physics library.

   Define a simple internal `Quantity` (if you want to avoid external dependencies) that stores a value and its unit.

3. NumPy and array support

   Ensure all functions (such as `KINETIC_ENERGY` or `Velocity`) are written to work with NumPy arrays as well (simple arithmetic often already works, but it is better to guarantee and document this).

## License

This project is provided as-is under the terms described by the repository
owner.
