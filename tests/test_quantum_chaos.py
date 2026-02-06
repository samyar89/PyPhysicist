import numpy as np
import pytest

from PyPhysicist.quantum_chaos import (
    band_structure,
    compare_to_gue,
    delta3_statistic,
    density_of_states,
    double_frequency_potential,
    floquet_operator,
    gue_delta3_prediction,
    gue_mean_spacing_ratio,
    inverse_participation_ratio,
    probability_density,
    quasi_periodic_potential,
    quantum_poincare_section,
    schrodinger_hamiltonian,
    select_contrasting_states,
    solve_schrodinger,
    spacing_statistics,
    wavefunction_contrast_data,
)


def test_quasi_periodic_potential_matches_phases_and_offset():
    x_grid = np.linspace(0.0, 2.0 * np.pi, 5)
    amplitudes = [1.0, 0.5]
    frequencies = [1.0, 2.0]
    offset = 0.2
    potential = quasi_periodic_potential(
        x_grid, amplitudes, frequencies, phases=None, offset=offset
    )
    explicit = quasi_periodic_potential(
        x_grid, amplitudes, frequencies, phases=[0.0, 0.0], offset=offset
    )
    assert np.allclose(potential, explicit)
    assert np.allclose(
        potential - offset,
        quasi_periodic_potential(x_grid, amplitudes, frequencies, phases=[0.0, 0.0]),
    )


def test_quasi_periodic_potential_shape_errors():
    x_grid = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError):
        quasi_periodic_potential(x_grid, [1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        quasi_periodic_potential(x_grid, [1.0, 2.0], [1.0, 2.0], phases=[0.0])


def test_double_frequency_potential_matches_helper():
    x_grid = np.linspace(0.0, 1.0, 4)
    potential = double_frequency_potential(x_grid, 1.2, 0.3, frequency_ratio=0.7)
    expected = quasi_periodic_potential(
        x_grid,
        [1.2, 0.3],
        [1.0, 0.7],
        phases=[0.0, 0.0],
    )
    assert np.allclose(potential, expected)


def test_schrodinger_hamiltonian_validation():
    x_grid = np.array([0.0, 0.4, 1.0])
    potential = np.zeros_like(x_grid)
    with pytest.raises(ValueError):
        schrodinger_hamiltonian(x_grid, potential)


def test_solve_schrodinger_sorted_and_orthonormal():
    x_grid = np.linspace(0.0, 2.0 * np.pi, 6)
    potential = np.zeros_like(x_grid)
    energies, states = solve_schrodinger(
        x_grid, potential, n_eigs=3, boundary="dirichlet"
    )
    assert np.all(np.diff(energies) >= 0.0)
    overlap = states.T.conjugate() @ states
    assert np.allclose(overlap, np.eye(3), atol=1e-8)


def test_band_structure_shape():
    x_grid = np.linspace(0.0, 1.0, 5)
    potential = np.zeros_like(x_grid)
    k_values = np.linspace(-np.pi, np.pi, 3)
    bands = band_structure(x_grid, potential, k_values, n_bands=2)
    assert bands.energies.shape == (len(k_values), 2)
    assert np.allclose(bands.k_values, k_values)


def test_density_of_states_positive():
    energies = np.array([0.0, 1.0, 2.0])
    dos = density_of_states(energies, n_points=16)
    assert dos.dos.shape == (16,)
    assert np.all(dos.dos >= 0.0)


def test_floquet_operator_unitary_for_diagonal():
    h1 = np.diag([0.0, 1.0])
    h2 = np.diag([0.2, -0.3])
    unitary = floquet_operator([h1, h2], [0.5, 0.25])
    identity = unitary.conjugate().T @ unitary
    assert np.allclose(identity, np.eye(2), atol=1e-10)


def test_quantum_poincare_section_stationary_state():
    floquet = np.eye(2, dtype=complex)
    initial_state = np.array([1.0, 0.0], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    section = quantum_poincare_section(
        floquet, initial_state, (sigma_z, sigma_x), n_steps=3
    )
    assert section.points.shape == (3, 2)
    assert np.allclose(section.points[:, 0], 1.0)
    assert np.allclose(section.points[:, 1], 0.0)


def test_probability_ipr_and_contrast_selection():
    state_localized = np.array([1.0, 0.0, 0.0])
    state_uniform = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    states = np.column_stack([state_uniform, state_localized])
    densities = probability_density(states)
    assert np.allclose(densities.sum(axis=0), 1.0)
    ipr = inverse_participation_ratio(states)
    assert ipr[1] > ipr[0]
    regular, chaotic = select_contrasting_states(states, n_pairs=1)
    assert np.allclose(regular[:, 0], state_uniform)
    assert np.allclose(chaotic[:, 0], state_localized)
    _, regular_density, chaotic_density = wavefunction_contrast_data(states[:, 0], states)
    assert regular_density.shape == chaotic_density.shape


def test_spacing_delta3_and_gue_helpers():
    energies = np.linspace(0.0, 5.0, 6)
    spacings, ratios = spacing_statistics(energies, degree=2)
    assert spacings.shape == (5,)
    assert ratios.shape == (4,)
    delta3, lengths = delta3_statistic(energies, lengths=[1, 2, 6])
    assert np.isnan(delta3[0])
    assert not np.isnan(delta3[1])
    assert np.isnan(delta3[2])
    stats = compare_to_gue(energies, lengths=[2, 3])
    assert stats.spacings.shape == (5,)
    assert stats.delta3.shape == (2,)
    prediction = gue_delta3_prediction([2.0, 3.0])
    assert prediction.shape == (2,)
    assert isinstance(gue_mean_spacing_ratio(), float)
