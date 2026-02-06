"""Quantum chaos utilities for quasi-periodic systems and spectral analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class BandStructure:
    """Container for band structure data."""

    k_values: ArrayLike
    energies: ArrayLike


@dataclass(frozen=True)
class DensityOfStates:
    """Container for density of states data."""

    energy_grid: ArrayLike
    dos: ArrayLike


@dataclass(frozen=True)
class PoincareSection:
    """Container for quantum Poincaré section data."""

    points: ArrayLike


@dataclass(frozen=True)
class RMTStatistics:
    """Container for random matrix theory statistics."""

    unfolded_levels: ArrayLike
    spacings: ArrayLike
    spacing_ratios: ArrayLike
    delta3: ArrayLike
    delta3_lengths: ArrayLike


def quasi_periodic_potential(
    x_grid: ArrayLike,
    amplitudes: Sequence[float],
    frequencies: Sequence[float],
    phases: Optional[Sequence[float]] = None,
    offset: float = 0.0,
) -> ArrayLike:
    """Construct a quasi-periodic potential with multiple incommensurate frequencies.

    Args:
        x_grid: Spatial grid.
        amplitudes: Potential amplitudes for each frequency.
        frequencies: Frequencies (wave numbers) for each component.
        phases: Optional phases for each component.
        offset: Constant offset.

    Returns:
        Potential evaluated on the grid.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if amplitudes.shape != frequencies.shape:
        raise ValueError("amplitudes and frequencies must have the same shape")
    if phases is None:
        phases = np.zeros_like(frequencies)
    phases = np.asarray(phases, dtype=float)
    if phases.shape != frequencies.shape:
        raise ValueError("phases must have the same shape as frequencies")
    argument = np.outer(frequencies, x_grid) + phases[:, None]
    return offset + (amplitudes[:, None] * np.cos(argument)).sum(axis=0)


def double_frequency_potential(
    x_grid: ArrayLike,
    amplitude_primary: float,
    amplitude_secondary: float,
    frequency_ratio: float = (math.sqrt(5) - 1) / 2,
    phase_secondary: float = 0.0,
) -> ArrayLike:
    """Convenience helper for a double-frequency quasi-periodic potential."""
    frequencies = np.array([1.0, frequency_ratio], dtype=float)
    amplitudes = np.array([amplitude_primary, amplitude_secondary], dtype=float)
    phases = np.array([0.0, phase_secondary], dtype=float)
    return quasi_periodic_potential(x_grid, amplitudes, frequencies, phases)


def _finite_difference_kinetic(
    n: int,
    dx: float,
    mass: float,
    hbar: float,
    boundary: str,
    bloch_phase: complex,
) -> ArrayLike:
    prefactor = -(hbar**2) / (2.0 * mass * dx**2)
    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)
    kinetic = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    if boundary == "periodic":
        kinetic[0, -1] = bloch_phase.conjugate()
        kinetic[-1, 0] = bloch_phase
    elif boundary != "dirichlet":
        raise ValueError("boundary must be 'periodic' or 'dirichlet'")
    return prefactor * kinetic


def schrodinger_hamiltonian(
    x_grid: ArrayLike,
    potential: ArrayLike,
    mass: float = 1.0,
    hbar: float = 1.0,
    boundary: str = "periodic",
    bloch_phase: complex = 1.0 + 0.0j,
) -> ArrayLike:
    """Build a finite-difference Hamiltonian matrix for 1D Schrödinger equation."""
    x_grid = np.asarray(x_grid, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if x_grid.ndim != 1:
        raise ValueError("x_grid must be 1D")
    if potential.shape != x_grid.shape:
        raise ValueError("potential must have the same shape as x_grid")
    if len(x_grid) < 3:
        raise ValueError("x_grid must have at least 3 points")
    dx = float(x_grid[1] - x_grid[0])
    if not np.allclose(np.diff(x_grid), dx):
        raise ValueError("x_grid must be uniformly spaced")
    kinetic = _finite_difference_kinetic(
        len(x_grid), dx, mass, hbar, boundary, bloch_phase
    )
    return kinetic + np.diag(potential)


def solve_schrodinger(
    x_grid: ArrayLike,
    potential: ArrayLike,
    n_eigs: Optional[int] = None,
    mass: float = 1.0,
    hbar: float = 1.0,
    boundary: str = "periodic",
    bloch_phase: complex = 1.0 + 0.0j,
) -> Tuple[ArrayLike, ArrayLike]:
    """Solve the 1D Schrödinger equation for eigenvalues and eigenvectors."""
    hamiltonian = schrodinger_hamiltonian(
        x_grid,
        potential,
        mass=mass,
        hbar=hbar,
        boundary=boundary,
        bloch_phase=bloch_phase,
    )
    energies, states = np.linalg.eigh(hamiltonian)
    if n_eigs is not None:
        return energies[:n_eigs], states[:, :n_eigs]
    return energies, states


def band_structure(
    x_grid: ArrayLike,
    potential: ArrayLike,
    k_values: ArrayLike,
    n_bands: Optional[int] = None,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> BandStructure:
    """Compute band structure by scanning Bloch phases over k-values."""
    x_grid = np.asarray(x_grid, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    length = float(x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0]))
    energies = []
    for k in k_values:
        phase = np.exp(1j * k * length)
        eigvals, _ = solve_schrodinger(
            x_grid,
            potential,
            n_eigs=n_bands,
            mass=mass,
            hbar=hbar,
            boundary="periodic",
            bloch_phase=phase,
        )
        energies.append(eigvals)
    energies = np.array(energies)
    return BandStructure(k_values=k_values, energies=energies)


def density_of_states(
    energies: ArrayLike,
    n_points: int = 512,
    bandwidth: Optional[float] = None,
) -> DensityOfStates:
    """Estimate density of states via Gaussian broadening."""
    energies = np.asarray(energies, dtype=float).ravel()
    emin, emax = energies.min(), energies.max()
    energy_grid = np.linspace(emin, emax, n_points)
    if bandwidth is None:
        bandwidth = 0.05 * (emax - emin) if emax > emin else 1.0
    diffs = energy_grid[:, None] - energies[None, :]
    kernel = np.exp(-0.5 * (diffs / bandwidth) ** 2)
    dos = kernel.sum(axis=1) / (math.sqrt(2 * math.pi) * bandwidth)
    return DensityOfStates(energy_grid=energy_grid, dos=dos)


def floquet_operator(
    hamiltonians: Sequence[ArrayLike],
    time_steps: Sequence[float],
    hbar: float = 1.0,
) -> ArrayLike:
    """Build a Floquet operator from piecewise-constant Hamiltonians."""
    if len(hamiltonians) != len(time_steps):
        raise ValueError("hamiltonians and time_steps must have the same length")
    unitary = np.eye(hamiltonians[0].shape[0], dtype=complex)
    for hamiltonian, dt in zip(hamiltonians, time_steps):
        evals, evecs = np.linalg.eigh(hamiltonian)
        phase = np.exp(-1j * evals * dt / hbar)
        unitary_step = evecs @ np.diag(phase) @ evecs.conjugate().T
        unitary = unitary_step @ unitary
    return unitary


def quantum_poincare_section(
    floquet: ArrayLike,
    initial_state: ArrayLike,
    observables: Tuple[ArrayLike, ArrayLike],
    n_steps: int,
) -> PoincareSection:
    """Generate a quantum Poincaré section from a Floquet operator."""
    state = np.asarray(initial_state, dtype=complex)
    state = state / np.linalg.norm(state)
    observable_a, observable_b = observables
    points = np.zeros((n_steps, 2), dtype=float)
    for idx in range(n_steps):
        expectation_a = np.vdot(state, observable_a @ state).real
        expectation_b = np.vdot(state, observable_b @ state).real
        points[idx] = (expectation_a, expectation_b)
        state = floquet @ state
        state = state / np.linalg.norm(state)
    return PoincareSection(points=points)


def probability_density(states: ArrayLike) -> ArrayLike:
    """Compute probability densities for eigenstates."""
    states = np.asarray(states)
    return np.abs(states) ** 2


def inverse_participation_ratio(states: ArrayLike) -> ArrayLike:
    """Compute inverse participation ratio to quantify localization."""
    densities = probability_density(states)
    return np.sum(densities**2, axis=0)


def select_contrasting_states(
    states: ArrayLike,
    n_pairs: int = 1,
) -> Tuple[ArrayLike, ArrayLike]:
    """Select regular and chaotic-like states based on IPR extremes."""
    ipr = inverse_participation_ratio(states)
    order = np.argsort(ipr)
    regular = states[:, order[:n_pairs]]
    chaotic = states[:, order[-n_pairs:]]
    return regular, chaotic


def _polynomial_unfolding(energies: ArrayLike, degree: int) -> ArrayLike:
    energies = np.asarray(energies, dtype=float)
    levels = np.sort(energies)
    n = len(levels)
    cumulative = np.arange(1, n + 1)
    coeffs = np.polyfit(levels, cumulative, degree)
    return np.polyval(coeffs, levels)


def spacing_statistics(energies: ArrayLike, degree: int = 5) -> Tuple[ArrayLike, ArrayLike]:
    """Compute unfolded spacings and spacing ratios."""
    unfolded = _polynomial_unfolding(energies, degree)
    spacings = np.diff(unfolded)
    ratios = np.minimum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
    return spacings, ratios


def delta3_statistic(
    energies: ArrayLike,
    lengths: Iterable[int],
    degree: int = 5,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute Dyson-Mehta Delta3 statistic for given window lengths."""
    unfolded = _polynomial_unfolding(energies, degree)
    levels = np.sort(unfolded)
    lengths = np.array(list(lengths), dtype=int)
    delta3_values = np.zeros_like(lengths, dtype=float)
    for idx, length in enumerate(lengths):
        if length < 2 or length >= len(levels):
            delta3_values[idx] = np.nan
            continue
        variances = []
        for start in range(len(levels) - length):
            window = levels[start : start + length]
            x = window - window[0]
            y = np.arange(length)
            coeffs = np.polyfit(x, y, 1)
            fit = np.polyval(coeffs, x)
            variances.append(np.mean((y - fit) ** 2))
        delta3_values[idx] = float(np.mean(variances))
    return delta3_values, lengths


def compare_to_gue(
    energies: ArrayLike,
    lengths: Iterable[int],
    degree: int = 5,
) -> RMTStatistics:
    """Compute RMT statistics for comparison with GUE predictions."""
    unfolded = _polynomial_unfolding(energies, degree)
    spacings = np.diff(unfolded)
    ratios = np.minimum(spacings[1:] / spacings[:-1], spacings[:-1] / spacings[1:])
    delta3_values, delta3_lengths = delta3_statistic(energies, lengths, degree)
    return RMTStatistics(
        unfolded_levels=unfolded,
        spacings=spacings,
        spacing_ratios=ratios,
        delta3=delta3_values,
        delta3_lengths=delta3_lengths,
    )


def gue_delta3_prediction(lengths: Iterable[float]) -> ArrayLike:
    """Analytical GUE Delta3 prediction for large lengths."""
    lengths = np.asarray(list(lengths), dtype=float)
    gamma = 0.5772156649
    return (1.0 / (math.pi**2)) * (np.log(2 * math.pi * lengths) + gamma - 1.25)


def gue_mean_spacing_ratio() -> float:
    """Mean spacing ratio for GUE statistics."""
    return 0.60266


def wavefunction_contrast_data(
    x_grid: ArrayLike,
    states: ArrayLike,
    n_pairs: int = 1,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Prepare wavefunction densities for regular vs chaotic comparisons."""
    regular, chaotic = select_contrasting_states(states, n_pairs=n_pairs)
    regular_density = probability_density(regular)
    chaotic_density = probability_density(chaotic)
    return x_grid, regular_density, chaotic_density
