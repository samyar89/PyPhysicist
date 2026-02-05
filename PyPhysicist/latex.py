"""LaTeX rendering utilities for PyPhysicist formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .electromagnetism.circuits import (
    current,
    resistance,
    resistance_parallel,
    resistance_series,
    voltage,
)
from .electromagnetism.electrostatics import capacitance, coulomb_force, electric_field
from .mechanics.dynamics import (
    centripetal_force,
    force,
    momentum,
    newton_second_law,
    weight,
)
from .mechanics.energy import (
    gravitational_potential_energy,
    kinetic_energy,
    mechanical_energy,
    spring_potential_energy,
    work,
)
from .mechanics.kinematics import centripetal_acceleration, velocity
from .relativity.gravity import schwarzschild_radius
from .relativity.special import length_contraction, relativistic_energy, time_dilation
from .thermodynamics.heat import entropy_change, heat_capacity
from .thermodynamics.ideal_gases import ideal_gas_pressure
from .units import Quantity
from .waves_optics.optics import refractive_index
from .waves_optics.waves import frequency, wave_power, wavelength


def _format_number(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value)
    if isinstance(value, (np.integer, np.floating)):
        return f"{value:g}"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, Quantity):
        return f"{_format_number(value.value)}\\,\\text{{{value.unit}}}"
    if isinstance(value, np.ndarray):
        return np.array2string(value)
    return _format_number(value)


@dataclass(frozen=True)
class LatexFormula:
    """Representation of a physics formula with LaTeX rendering helpers."""

    name: str
    result_symbol: str
    rhs_template: str
    parameters: tuple[str, ...]
    parameter_symbols: Mapping[str, str]
    evaluator: Callable[..., Any] | None = None
    variadic_param: str | None = None
    variadic_format: str | None = None

    def symbolic_equation(self) -> str:
        """Return the symbolic LaTeX equation."""
        return self._equation(self._symbol_mapping())

    def substitute_equation(self, values: Mapping[str, Any]) -> str:
        """Return the LaTeX equation with substituted values."""
        return self._equation(self._value_mapping(values))

    def calculate(self, **values: Any) -> "LatexCalculation":
        """Compute the numeric result and package it with the formula."""
        if self.evaluator is None:
            raise ValueError(f"Formula '{self.name}' does not have an evaluator.")
        if self.variadic_param:
            variadic_values = values.get(self.variadic_param)
            if variadic_values is None:
                raise ValueError(
                    f"Formula '{self.name}' requires '{self.variadic_param}' values."
                )
            if not isinstance(variadic_values, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"'{self.variadic_param}' must be a list or tuple of values."
                )
            result = self.evaluator(*variadic_values)
        else:
            result = self.evaluator(**values)
        return LatexCalculation(formula=self, values=values, result=result)

    def _equation(self, mapping: Mapping[str, str]) -> str:
        rhs = self.rhs_template.format_map(mapping)
        return f"{self.result_symbol} = {rhs}"

    def _symbol_mapping(self) -> dict[str, str]:
        mapping = {param: self.parameter_symbols.get(param, param) for param in self.parameters}
        if self.variadic_param:
            mapping[self.variadic_param] = self._format_variadic(symbolic=True, values=None)
        return mapping

    def _value_mapping(self, values: Mapping[str, Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for param in self.parameters:
            if param == self.variadic_param:
                mapping[param] = self._format_variadic(symbolic=False, values=values.get(param))
            else:
                if param not in values:
                    raise ValueError(f"Missing value for '{param}' in formula '{self.name}'.")
                mapping[param] = _format_value(values[param])
        return mapping

    def _format_variadic(self, symbolic: bool, values: Iterable[Any] | None) -> str:
        if symbolic:
            if self.variadic_format == "inverse_sum":
                return " + ".join(
                    [
                        "\\frac{1}{R_1}",
                        "\\frac{1}{R_2}",
                        "\\cdots",
                        "\\frac{1}{R_n}",
                    ]
                )
            return "R_1 + R_2 + \\cdots + R_n"
        if values is None:
            raise ValueError(f"Missing values for '{self.variadic_param}'.")
        formatted_values = [_format_value(value) for value in values]
        if self.variadic_format == "inverse_sum":
            formatted_values = [f"\\frac{{1}}{{{value}}}" for value in formatted_values]
        return " + ".join(formatted_values)


@dataclass(frozen=True)
class LatexCalculation:
    """Result of a numeric calculation that can be rendered as LaTeX."""

    formula: LatexFormula
    values: Mapping[str, Any]
    result: Any

    def to_latex(self) -> str:
        equation = self.formula.substitute_equation(self.values)
        return f"{equation} = {_format_value(self.result)}"


class LatexConverter:
    """Central access point for LaTeX conversions of PyPhysicist formulas."""

    def __init__(self, formulas: Mapping[str, LatexFormula] | None = None) -> None:
        self._formulas = dict(formulas or DEFAULT_FORMULAS)

    def register(self, formula: LatexFormula) -> None:
        self._formulas[formula.name] = formula

    def formula(self, name: str) -> LatexFormula:
        if name not in self._formulas:
            raise KeyError(f"Unknown formula '{name}'.")
        return self._formulas[name]

    def symbolic(self, name: str) -> str:
        return self.formula(name).symbolic_equation()

    def substitute(self, name: str, **values: Any) -> str:
        return self.formula(name).substitute_equation(values)

    def calculate(self, name: str, **values: Any) -> LatexCalculation:
        return self.formula(name).calculate(**values)


DEFAULT_FORMULAS: dict[str, LatexFormula] = {
    "velocity": LatexFormula(
        name="velocity",
        result_symbol="v",
        rhs_template=r"\\frac{{{distance}}}{{{time}}}",
        parameters=("distance", "time"),
        parameter_symbols={"distance": "d", "time": "t"},
        evaluator=velocity,
    ),
    "centripetal_acceleration": LatexFormula(
        name="centripetal_acceleration",
        result_symbol="a_c",
        rhs_template=r"\\frac{{{speed}}^2}{{{radius}}}",
        parameters=("speed", "radius"),
        parameter_symbols={"speed": "v", "radius": "r"},
        evaluator=centripetal_acceleration,
    ),
    "force": LatexFormula(
        name="force",
        result_symbol="F",
        rhs_template=r"{mass}\\, {acceleration}",
        parameters=("mass", "acceleration"),
        parameter_symbols={"mass": "m", "acceleration": "a"},
        evaluator=force,
    ),
    "newton_second_law": LatexFormula(
        name="newton_second_law",
        result_symbol="F",
        rhs_template=r"{mass}\\, {acceleration}",
        parameters=("mass", "acceleration"),
        parameter_symbols={"mass": "m", "acceleration": "a"},
        evaluator=newton_second_law,
    ),
    "momentum": LatexFormula(
        name="momentum",
        result_symbol="p",
        rhs_template=r"{mass}\\, {velocity}",
        parameters=("mass", "velocity"),
        parameter_symbols={"mass": "m", "velocity": "v"},
        evaluator=momentum,
    ),
    "centripetal_force": LatexFormula(
        name="centripetal_force",
        result_symbol="F_c",
        rhs_template=r"\\frac{{{mass}\\, {speed}^2}}{{{radius}}}",
        parameters=("mass", "speed", "radius"),
        parameter_symbols={"mass": "m", "speed": "v", "radius": "r"},
        evaluator=centripetal_force,
    ),
    "weight": LatexFormula(
        name="weight",
        result_symbol="W",
        rhs_template=r"{mass}\\, {gravity}",
        parameters=("mass", "gravity"),
        parameter_symbols={"mass": "m", "gravity": "g"},
        evaluator=weight,
    ),
    "kinetic_energy": LatexFormula(
        name="kinetic_energy",
        result_symbol="K",
        rhs_template=r"\\frac{1}{2} {mass}\\, {velocity}^2",
        parameters=("mass", "velocity"),
        parameter_symbols={"mass": "m", "velocity": "v"},
        evaluator=kinetic_energy,
    ),
    "gravitational_potential_energy": LatexFormula(
        name="gravitational_potential_energy",
        result_symbol="U_g",
        rhs_template=r"{mass}\\, {gravity}\\, {height}",
        parameters=("mass", "gravity", "height"),
        parameter_symbols={"mass": "m", "gravity": "g", "height": "h"},
        evaluator=gravitational_potential_energy,
    ),
    "mechanical_energy": LatexFormula(
        name="mechanical_energy",
        result_symbol="E",
        rhs_template=r"{kinetic} + {potential}",
        parameters=("kinetic", "potential"),
        parameter_symbols={"kinetic": "K", "potential": "U"},
        evaluator=mechanical_energy,
    ),
    "spring_potential_energy": LatexFormula(
        name="spring_potential_energy",
        result_symbol="U_s",
        rhs_template=r"\\frac{1}{2} {spring_constant}\\, {displacement}^2",
        parameters=("spring_constant", "displacement"),
        parameter_symbols={"spring_constant": "k", "displacement": "x"},
        evaluator=spring_potential_energy,
    ),
    "work": LatexFormula(
        name="work",
        result_symbol="W",
        rhs_template=r"{force_value}\\, {displacement}",
        parameters=("force_value", "displacement"),
        parameter_symbols={"force_value": "F", "displacement": "d"},
        evaluator=work,
    ),
    "coulomb_force": LatexFormula(
        name="coulomb_force",
        result_symbol="F",
        rhs_template=r"\\frac{{k\\, {charge1}\\, {charge2}}}{{{distance}^2}}",
        parameters=("charge1", "charge2", "distance"),
        parameter_symbols={"charge1": "q_1", "charge2": "q_2", "distance": "r"},
        evaluator=coulomb_force,
    ),
    "electric_field": LatexFormula(
        name="electric_field",
        result_symbol="E",
        rhs_template=r"\\frac{{{force_value}}}{{{charge}}}",
        parameters=("force_value", "charge"),
        parameter_symbols={"force_value": "F", "charge": "q"},
        evaluator=electric_field,
    ),
    "capacitance": LatexFormula(
        name="capacitance",
        result_symbol="C",
        rhs_template=r"\\frac{{{charge}}}{{{voltage}}}",
        parameters=("charge", "voltage"),
        parameter_symbols={"charge": "Q", "voltage": "V"},
        evaluator=capacitance,
    ),
    "voltage": LatexFormula(
        name="voltage",
        result_symbol="V",
        rhs_template=r"{current}\\, {resistance}",
        parameters=("current", "resistance"),
        parameter_symbols={"current": "I", "resistance": "R"},
        evaluator=voltage,
    ),
    "current": LatexFormula(
        name="current",
        result_symbol="I",
        rhs_template=r"\\frac{{{voltage_value}}}{{{resistance}}}",
        parameters=("voltage_value", "resistance"),
        parameter_symbols={"voltage_value": "V", "resistance": "R"},
        evaluator=current,
    ),
    "resistance": LatexFormula(
        name="resistance",
        result_symbol="R",
        rhs_template=r"\\frac{{{voltage_value}}}{{{current_value}}}",
        parameters=("voltage_value", "current_value"),
        parameter_symbols={"voltage_value": "V", "current_value": "I"},
        evaluator=resistance,
    ),
    "resistance_series": LatexFormula(
        name="resistance_series",
        result_symbol="R_{eq}",
        rhs_template=r"{resistances}",
        parameters=("resistances",),
        parameter_symbols={"resistances": "R_1 + R_2 + \\cdots + R_n"},
        evaluator=resistance_series,
        variadic_param="resistances",
        variadic_format="sum",
    ),
    "resistance_parallel": LatexFormula(
        name="resistance_parallel",
        result_symbol="R_{eq}",
        rhs_template=r"\\left( {resistances} \\right)^{-1}",
        parameters=("resistances",),
        parameter_symbols={"resistances": "\\frac{1}{R_1} + \\frac{1}{R_2} + \\cdots + \\frac{1}{R_n}"},
        evaluator=resistance_parallel,
        variadic_param="resistances",
        variadic_format="inverse_sum",
    ),
    "refractive_index": LatexFormula(
        name="refractive_index",
        result_symbol="n",
        rhs_template=r"\\frac{{{speed_of_light}}}{{{medium_speed}}}",
        parameters=("speed_of_light", "medium_speed"),
        parameter_symbols={"speed_of_light": "c", "medium_speed": "v"},
        evaluator=refractive_index,
    ),
    "frequency": LatexFormula(
        name="frequency",
        result_symbol="f",
        rhs_template=r"\\frac{{{wave_speed}}}{{{wavelength_value}}}",
        parameters=("wave_speed", "wavelength_value"),
        parameter_symbols={"wave_speed": "v", "wavelength_value": "\\lambda"},
        evaluator=frequency,
    ),
    "wavelength": LatexFormula(
        name="wavelength",
        result_symbol="\\lambda",
        rhs_template=r"\\frac{{{wave_speed}}}{{{frequency_hz}}}",
        parameters=("wave_speed", "frequency_hz"),
        parameter_symbols={"wave_speed": "v", "frequency_hz": "f"},
        evaluator=wavelength,
    ),
    "wave_power": LatexFormula(
        name="wave_power",
        result_symbol="P",
        rhs_template=r"\\frac{{{energy}}}{{{time}}}",
        parameters=("energy", "time"),
        parameter_symbols={"energy": "E", "time": "t"},
        evaluator=wave_power,
    ),
    "heat_capacity": LatexFormula(
        name="heat_capacity",
        result_symbol="c",
        rhs_template=r"\\frac{{{heat}}}{{{mass}\\, {delta_t}}}",
        parameters=("heat", "mass", "delta_t"),
        parameter_symbols={"heat": "Q", "mass": "m", "delta_t": "\\Delta T"},
        evaluator=heat_capacity,
    ),
    "entropy_change": LatexFormula(
        name="entropy_change",
        result_symbol="\\Delta S",
        rhs_template=r"\\frac{{{heat}}}{{{temperature}}}",
        parameters=("heat", "temperature"),
        parameter_symbols={"heat": "Q", "temperature": "T"},
        evaluator=entropy_change,
    ),
    "ideal_gas_pressure": LatexFormula(
        name="ideal_gas_pressure",
        result_symbol="P",
        rhs_template=r"\\frac{{{n}\\, {r}\\, {t}}}{{{v}}}",
        parameters=("n", "r", "t", "v"),
        parameter_symbols={"n": "n", "r": "R", "t": "T", "v": "V"},
        evaluator=ideal_gas_pressure,
    ),
    "time_dilation": LatexFormula(
        name="time_dilation",
        result_symbol="t",
        rhs_template=r"\\frac{{{proper_time}}}{{\\sqrt{1 - \\frac{{{velocity}^2}}{{{c}^2}}}}}",
        parameters=("proper_time", "velocity", "c"),
        parameter_symbols={"proper_time": "t_0", "velocity": "v", "c": "c"},
        evaluator=time_dilation,
    ),
    "length_contraction": LatexFormula(
        name="length_contraction",
        result_symbol="L",
        rhs_template=r"{proper_length}\\, \\sqrt{1 - \\frac{{{velocity}^2}}{{{c}^2}}}",
        parameters=("proper_length", "velocity", "c"),
        parameter_symbols={"proper_length": "L_0", "velocity": "v", "c": "c"},
        evaluator=length_contraction,
    ),
    "relativistic_energy": LatexFormula(
        name="relativistic_energy",
        result_symbol="E",
        rhs_template=r"{mass}\\, {c}^2",
        parameters=("mass", "c"),
        parameter_symbols={"mass": "m", "c": "c"},
        evaluator=relativistic_energy,
    ),
    "schwarzschild_radius": LatexFormula(
        name="schwarzschild_radius",
        result_symbol="r_s",
        rhs_template=r"\\frac{{2\\, {g}\\, {mass}}}{{{c}^2}}",
        parameters=("mass", "g", "c"),
        parameter_symbols={"mass": "M", "g": "G", "c": "c"},
        evaluator=schwarzschild_radius,
    ),
}

__all__ = [
    "LatexFormula",
    "LatexCalculation",
    "LatexConverter",
    "DEFAULT_FORMULAS",
]
