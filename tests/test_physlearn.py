import numpy as np

from PyPhysicist.physlearn import PhysLearn


def test_physlearn_train_predict():
    x = np.linspace(0, 1, 10)[:, None]
    y = 2 * x + 1
    pl = PhysLearn(equation="heat", domain="1d")
    pl.build_surrogate("linear")
    pl.train({"x": x, "y": y})
    pred = pl.predict(x)
    assert pred.mean.shape == y.shape
    assert np.allclose(pred.mean, y, atol=1e-6)


def test_physlearn_suggest_experiment():
    x = np.linspace(0, 1, 6)[:, None]
    y = 3 * x
    pl = PhysLearn()
    pl.build_surrogate("ensemble", ensemble_size=3)
    pl.train({"x": x, "y": y})
    candidates = np.linspace(0, 1, 5)[:, None]
    result = pl.suggest_experiment("bald", candidates, budget=2)
    assert result.selected_indices.shape == (2,)


def test_physlearn_assimilate():
    observations = np.array([1.0, 2.0, 3.0])
    pl = PhysLearn()
    result = pl.assimilate(observations, ensemble_size=5)
    assert result.analysis_mean.shape == observations.shape
