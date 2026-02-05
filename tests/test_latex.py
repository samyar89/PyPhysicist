import pytest

import PyPhysicist as pp


def test_symbolic_formula():
    converter = pp.LatexConverter()
    assert converter.symbolic("velocity") == r"v = \\frac{d}{t}"


def test_substitute_formula_with_quantity():
    converter = pp.LatexConverter()
    mass = pp.Quantity(2, "kg")
    equation = converter.substitute("force", mass=mass, acceleration=3)
    assert equation == r"F = 2\,\text{kg}\\, 3"


def test_calculate_formula_to_latex():
    converter = pp.LatexConverter()
    calculation = converter.calculate("velocity", distance=10, time=4)
    assert calculation.to_latex() == r"v = \\frac{10}{4} = 2.5"


def test_variadic_symbolic_equations():
    converter = pp.LatexConverter()
    assert converter.symbolic("resistance_series") == r"R_{eq} = R_1 + R_2 + \cdots + R_n"
    assert (
        converter.symbolic("resistance_parallel")
        == r"R_{eq} = \\left( \frac{1}{R_1} + \frac{1}{R_2} + \cdots + \frac{1}{R_n} \\right)^{-1}"
    )


def test_variadic_substitution_equations():
    converter = pp.LatexConverter()
    equation = converter.substitute("resistance_parallel", resistances=[2, 4, 8])
    assert equation == r"R_{eq} = \\left( \frac{1}{2} + \frac{1}{4} + \frac{1}{8} \\right)^{-1}"


def test_missing_values_raise():
    converter = pp.LatexConverter()
    with pytest.raises(ValueError, match="Missing value for 'time'"):
        converter.substitute("velocity", distance=5)
    with pytest.raises(ValueError, match="requires 'resistances' values"):
        converter.calculate("resistance_series")


def test_unknown_formula_raise():
    converter = pp.LatexConverter()
    with pytest.raises(KeyError, match="Unknown formula"):
        converter.symbolic("unknown")
