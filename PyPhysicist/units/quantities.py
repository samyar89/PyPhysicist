"""Lightweight quantity representation for dimensional awareness.

This is intentionally minimal to keep dependencies light. For richer unit
support, consider integrating Pint and wrapping inputs before calling the
formula helpers.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class Quantity:
    """Numeric value paired with a unit string.

    Args:
        value: Scalar or array-like numeric value.
        unit: Unit string (e.g., "m/s", "J", "kg").
    """

    value: Any
    unit: str

    def as_array(self) -> np.ndarray:
        """Return the numeric value as a NumPy array."""
        return np.asarray(self.value)

    def __repr__(self) -> str:
        return f"Quantity(value={self.value!r}, unit='{self.unit}')"

    def _mul_div_unit(self, other: "Quantity", sign: int) -> tuple[np.ndarray, str]:
        from .conversion import UnitError, format_unit, parse_unit

        def to_base(value: Any, unit: str) -> tuple[np.ndarray, Dict[str, int]]:
            spec = parse_unit(unit)
            if spec.offset != 0.0:
                raise UnitError(
                    f"Offset unit '{unit}' cannot be used with multiplication/division."
                )
            return np.asarray(value) * spec.scale, dict(spec.dims)

        def merge_dims(
            dims: Dict[str, int], other_dims: Dict[str, int], multiplier: int
        ) -> Dict[str, int]:
            combined = dict(dims)
            for key, power in other_dims.items():
                combined[key] = combined.get(key, 0) + power * multiplier
                if combined[key] == 0:
                    del combined[key]
            return combined

        self_value, self_dims = to_base(self.value, self.unit)
        other_value, other_dims = to_base(other.value, other.unit)
        value = self_value * other_value if sign == 1 else self_value / other_value
        combined_dims = merge_dims(self_dims, other_dims, sign)
        return value, format_unit(combined_dims)

    def __add__(self, other: object) -> "Quantity":
        if isinstance(other, Quantity):
            from .conversion import convert_value

            other_value = convert_value(other.value, other.unit, self.unit)
            return Quantity(np.asarray(self.value) + other_value, self.unit)
        return NotImplemented

    def __radd__(self, other: object) -> "Quantity":
        return self.__add__(other)

    def __sub__(self, other: object) -> "Quantity":
        if isinstance(other, Quantity):
            from .conversion import convert_value

            other_value = convert_value(other.value, other.unit, self.unit)
            return Quantity(np.asarray(self.value) - other_value, self.unit)
        return NotImplemented

    def __rsub__(self, other: object) -> "Quantity":
        if isinstance(other, Quantity):
            return other.__sub__(self)
        return NotImplemented

    def __mul__(self, other: object) -> "Quantity":
        if isinstance(other, Quantity):
            value, unit = self._mul_div_unit(other, sign=1)
            return Quantity(value, unit)
        if np.isscalar(other):
            return Quantity(np.asarray(self.value) * other, self.unit)
        return NotImplemented

    def __rmul__(self, other: object) -> "Quantity":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "Quantity":
        if isinstance(other, Quantity):
            value, unit = self._mul_div_unit(other, sign=-1)
            return Quantity(value, unit)
        if np.isscalar(other):
            return Quantity(np.asarray(self.value) / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other: object) -> "Quantity":
        if np.isscalar(other):
            from .conversion import format_unit, parse_unit

            spec = parse_unit(self.unit)
            if spec.offset != 0.0:
                from .conversion import UnitError

                raise UnitError(
                    f"Offset unit '{self.unit}' cannot be used with multiplication/division."
                )
            inverted_dims = {key: -power for key, power in spec.dims.items()}
            value = np.asarray(other) / (np.asarray(self.value) * spec.scale)
            return Quantity(value, format_unit(inverted_dims))
        return NotImplemented

    def __neg__(self) -> "Quantity":
        return Quantity(-np.asarray(self.value), self.unit)

    def __pow__(self, power: object, modulo: object = None) -> "Quantity":
        if modulo is not None:
            return NotImplemented
        if not isinstance(power, (int, np.integer)):
            return NotImplemented
        from .conversion import UnitError, format_unit, parse_unit

        spec = parse_unit(self.unit)
        if spec.offset != 0.0:
            raise UnitError(
                f"Offset unit '{self.unit}' cannot be raised to a power."
            )
        value = (np.asarray(self.value) * spec.scale) ** power
        dims = {key: exponent * power for key, exponent in spec.dims.items()}
        return Quantity(value, format_unit(dims))


__all__ = ["Quantity"]
