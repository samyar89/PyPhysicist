"""Lightweight quantity representation for dimensional awareness.

This is intentionally minimal to keep dependencies light. For richer unit
support, consider integrating Pint and wrapping inputs before calling the
formula helpers.
"""

from dataclasses import dataclass
from typing import Any

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


__all__ = ["Quantity"]
