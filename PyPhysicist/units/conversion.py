"""Unit parsing, conversion, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .quantities import Quantity


class UnitError(ValueError):
    """Raised when units are incompatible or cannot be converted."""


@dataclass(frozen=True)
class UnitSpec:
    """Parsed unit specification."""

    scale: float
    dims: Dict[str, int]
    offset: float = 0.0


_BASE_UNITS = ("m", "kg", "s", "A", "K", "mol", "cd")


_UNIT_DEFINITIONS: Dict[str, UnitSpec] = {
    "1": UnitSpec(1.0, {}),
    "m": UnitSpec(1.0, {"m": 1}),
    "meter": UnitSpec(1.0, {"m": 1}),
    "meters": UnitSpec(1.0, {"m": 1}),
    "km": UnitSpec(1.0e3, {"m": 1}),
    "cm": UnitSpec(1.0e-2, {"m": 1}),
    "mm": UnitSpec(1.0e-3, {"m": 1}),
    "um": UnitSpec(1.0e-6, {"m": 1}),
    "nm": UnitSpec(1.0e-9, {"m": 1}),
    "s": UnitSpec(1.0, {"s": 1}),
    "sec": UnitSpec(1.0, {"s": 1}),
    "second": UnitSpec(1.0, {"s": 1}),
    "ms": UnitSpec(1.0e-3, {"s": 1}),
    "us": UnitSpec(1.0e-6, {"s": 1}),
    "ns": UnitSpec(1.0e-9, {"s": 1}),
    "min": UnitSpec(60.0, {"s": 1}),
    "hr": UnitSpec(3600.0, {"s": 1}),
    "kg": UnitSpec(1.0, {"kg": 1}),
    "g": UnitSpec(1.0e-3, {"kg": 1}),
    "mg": UnitSpec(1.0e-6, {"kg": 1}),
    "Mg": UnitSpec(1.0e3, {"kg": 1}),
    "A": UnitSpec(1.0, {"A": 1}),
    "K": UnitSpec(1.0, {"K": 1}),
    "degC": UnitSpec(1.0, {"K": 1}, offset=273.15),
    "mol": UnitSpec(1.0, {"mol": 1}),
    "cd": UnitSpec(1.0, {"cd": 1}),
    "rad": UnitSpec(1.0, {}),
    "Hz": UnitSpec(1.0, {"s": -1}),
    "N": UnitSpec(1.0, {"kg": 1, "m": 1, "s": -2}),
    "J": UnitSpec(1.0, {"kg": 1, "m": 2, "s": -2}),
    "W": UnitSpec(1.0, {"kg": 1, "m": 2, "s": -3}),
    "Pa": UnitSpec(1.0, {"kg": 1, "m": -1, "s": -2}),
    "C": UnitSpec(1.0, {"A": 1, "s": 1}),
    "V": UnitSpec(1.0, {"kg": 1, "m": 2, "s": -3, "A": -1}),
    "ohm": UnitSpec(1.0, {"kg": 1, "m": 2, "s": -3, "A": -2}),
    "F": UnitSpec(1.0, {"kg": -1, "m": -2, "s": 4, "A": 2}),
}


def _normalize_unit(unit: str) -> str:
    unit = unit.strip()
    if unit in {"", "1"}:
        return "1"
    unit = unit.replace("·", "*").replace("μ", "u").replace("µ", "u")
    unit = unit.replace("°C", "degC")
    unit = unit.replace(" ", "")
    return unit


def _split_unit_expression(unit: str) -> Tuple[Iterable[str], Iterable[str]]:
    parts = unit.split("/")
    numerator = parts[0]
    denominator = parts[1:] if len(parts) > 1 else []
    numerator_tokens = [token for token in numerator.split("*") if token]
    denominator_tokens = []
    for part in denominator:
        denominator_tokens.extend(token for token in part.split("*") if token)
    return numerator_tokens, denominator_tokens


def _parse_token(token: str) -> Tuple[str, int]:
    if "^" in token:
        symbol, exponent = token.split("^", 1)
    elif "**" in token:
        symbol, exponent = token.split("**", 1)
    else:
        symbol, exponent = token, "1"
    try:
        exp_value = int(exponent)
    except ValueError as exc:
        raise UnitError(f"Invalid exponent in unit '{token}'.") from exc
    return symbol, exp_value


def _merge_dims(target: Dict[str, int], dims: Dict[str, int], multiplier: int) -> None:
    for key, power in dims.items():
        target[key] = target.get(key, 0) + power * multiplier
        if target[key] == 0:
            del target[key]


def parse_unit(unit: str) -> UnitSpec:
    """Parse a unit expression into scale and dimensions."""
    unit = _normalize_unit(unit)
    if unit == "1":
        return UnitSpec(1.0, {})
    numerator_tokens, denominator_tokens = _split_unit_expression(unit)
    dims: Dict[str, int] = {}
    scale = 1.0
    offset_unit = None

    for token, sign in [(token, 1) for token in numerator_tokens] + [
        (token, -1) for token in denominator_tokens
    ]:
        symbol, exponent = _parse_token(token)
        if symbol not in _UNIT_DEFINITIONS:
            raise UnitError(f"Unknown unit '{symbol}'.")
        spec = _UNIT_DEFINITIONS[symbol]
        if spec.offset != 0.0:
            if exponent != 1 or len(numerator_tokens) + len(denominator_tokens) > 1 or sign == -1:
                raise UnitError(f"Offset unit '{symbol}' cannot be combined with other units.")
            offset_unit = spec
        scale *= spec.scale ** (exponent * sign)
        _merge_dims(dims, spec.dims, exponent * sign)

    if offset_unit:
        return UnitSpec(scale, dims, offset=offset_unit.offset)
    return UnitSpec(scale, dims)


def format_unit(dims: Dict[str, int]) -> str:
    """Format base unit dimensions as a string."""
    if not dims:
        return "1"
    numerator = []
    denominator = []
    for base in _BASE_UNITS:
        power = dims.get(base, 0)
        if power > 0:
            numerator.append(f"{base}" if power == 1 else f"{base}^{power}")
        elif power < 0:
            power = abs(power)
            denominator.append(f"{base}" if power == 1 else f"{base}^{power}")
    numerator_str = "*".join(numerator) if numerator else "1"
    denominator_str = "*".join(denominator)
    if denominator_str:
        return f"{numerator_str}/{denominator_str}"
    return numerator_str


def convert_value(
    value: float | np.ndarray,
    from_unit: str,
    to_unit: str,
    *,
    ignore_offset: bool = False,
) -> float | np.ndarray:
    """Convert numeric values between compatible units."""
    from_spec = parse_unit(from_unit)
    to_spec = parse_unit(to_unit)
    if from_spec.dims != to_spec.dims:
        raise UnitError(f"Incompatible units '{from_unit}' and '{to_unit}'.")
    value = np.asarray(value)
    if from_spec.offset and not ignore_offset:
        value = value + from_spec.offset
    value = value * (from_spec.scale / to_spec.scale)
    if to_spec.offset and not ignore_offset:
        value = value - to_spec.offset
    return value


def coerce_value(
    value,
    expected_unit: str | None,
    *,
    name: str,
    ignore_offset: bool = False,
) -> Tuple[np.ndarray, str | None]:
    """Coerce input to numeric array and normalize unit."""
    if isinstance(value, Quantity):
        unit = value.unit
        numeric = value.value
        if expected_unit is not None:
            numeric = convert_value(
                numeric, unit, expected_unit, ignore_offset=ignore_offset
            )
            unit = expected_unit
        return np.asarray(numeric), unit
    if expected_unit is None:
        return np.asarray(value), None
    return np.asarray(value), expected_unit


def ensure_same_units(unit_a: str, unit_b: str, *, name_a: str, name_b: str) -> None:
    """Ensure two unit strings describe the same dimensions."""
    spec_a = parse_unit(unit_a)
    spec_b = parse_unit(unit_b)
    if spec_a.dims != spec_b.dims:
        raise UnitError(
            f"Incompatible units for {name_a} ({unit_a}) and {name_b} ({unit_b})."
        )


def wrap_quantity(value: np.ndarray, unit: str, *inputs) -> Quantity | np.ndarray:
    """Return Quantity if any input is a Quantity."""
    if any(isinstance(item, Quantity) for item in inputs):
        return Quantity(value, unit)
    return value


__all__ = [
    "UnitError",
    "UnitSpec",
    "convert_value",
    "coerce_value",
    "ensure_same_units",
    "format_unit",
    "parse_unit",
    "wrap_quantity",
]
