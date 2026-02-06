import numpy as np

from PyPhysicist.physlearn import PhysLearn, discrepancy_dataset
from PyPhysicist.physlearn.active_learning import expected_improvement
from PyPhysicist.physlearn.data_assimilation import EnKF
from PyPhysicist.physlearn.datasets import burgers_1d, low_high_fidelity_pair


def test_registry_has_operator_models():
    pl = PhysLearn()
    models = pl.model_registry.list()
    assert "fno" in models
    assert "deeponet" in models
    assert "pinn" in models
    assert "hybrid" in models


def test_expected_improvement_shape():
    mean = np.array([1.0, 0.5, 0.2])
    std = np.array([0.1, 0.2, 0.3])
    scores = expected_improvement(mean, std, best=0.2)
    assert scores.shape == mean.shape


def test_cost_aware_suggestion():
    x = np.linspace(0, 1, 10)[:, None]
    y = 2 * x
    pl = PhysLearn()
    pl.build_surrogate("linear")
    pl.train({"x": x, "y": y})
    candidates = np.linspace(0, 1, 5)[:, None]
    costs = np.array([1.0, 0.5, 2.0, 1.5, 1.0])
    result = pl.suggest_experiment("variance", candidates, budget=2, costs=costs)
    assert result.selected_indices.shape == (2,)


def test_enkf_reduces_error():
    true_state = np.array([1.0, 2.0])
    ensemble = true_state + np.array([[1.0, -1.0], [0.5, -0.5], [1.5, -1.5]])
    observations = true_state + np.array([0.1, -0.1])
    enkf = EnKF()
    result = enkf.update(ensemble, observations, observation_noise=1e-3, inflation=1.0)
    forecast_mean = ensemble.mean(axis=0)
    analysis_mean = result.analysis_mean
    assert np.linalg.norm(analysis_mean - true_state) < np.linalg.norm(forecast_mean - true_state)


def test_discrepancy_dataset_shapes():
    low, high = low_high_fidelity_pair(burgers_1d, n_samples=16, seed=0)
    disc = discrepancy_dataset(low, high)
    assert disc["x"].shape == low["x"].shape
    assert disc["y"].shape == low["y"].shape
