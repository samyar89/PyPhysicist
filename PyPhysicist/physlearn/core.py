"""Core PhysLearn API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .active_learning import (
    AcquisitionResult,
    acquisition_loop,
    bald_acquisition,
    cost_aware_selection,
    expected_improvement,
    variance_acquisition,
)
from .data_assimilation import EnKF, EnKFResult
from .models.base import BaseSurrogate, SurrogatePrediction
from .models.deeponet import DeepONetSurrogate
from .models.fno import FNOOperatorSurrogate
from .models.hybrid import NeuralCorrectorSurrogate
from .models.pinn import PINNSurrogate
from .models.simple import EnsembleSurrogate, IdentitySurrogate, LinearSurrogate
from .multifidelity import MultiFidelityTrainer
from .registry import ModelRegistry, SolverRegistry


@dataclass
class PhysLearn:
    """High-level orchestrator for physics-informed learning."""

    equation: Optional[str] = None
    domain: Optional[Any] = None
    device: str = "cpu"
    seed: int = 42
    solver_registry: SolverRegistry = field(default_factory=SolverRegistry)
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    surrogate: Optional[BaseSurrogate] = None

    def __post_init__(self) -> None:
        np.random.seed(self.seed)
        default_builders = {
            "linear": lambda **_: LinearSurrogate(),
            "ensemble": self._build_ensemble,
            "identity": lambda **_: IdentitySurrogate(),
            "fno": lambda **kwargs: FNOOperatorSurrogate(config=kwargs),
            "deeponet": lambda **kwargs: DeepONetSurrogate(config=kwargs),
            "pinn": lambda **kwargs: PINNSurrogate(config=kwargs),
            "hybrid": lambda **kwargs: NeuralCorrectorSurrogate(**kwargs),
        }
        self.model_registry.ensure_default_models(default_builders)

    def _build_ensemble(self, **kwargs: Any) -> EnsembleSurrogate:
        size = int(kwargs.get("ensemble_size", 5))
        members = [LinearSurrogate() for _ in range(size)]
        return EnsembleSurrogate(members=members)

    def register_solver(self, name: str, solver_fn: Any) -> None:
        self.solver_registry.register(name, solver_fn)

    def build_surrogate(self, model: str, **kwargs: Any) -> BaseSurrogate:
        self.surrogate = self.model_registry.build(model, **kwargs)
        return self.surrogate

    def train(self, dataset: Dict[str, Any], **kwargs: Any) -> BaseSurrogate:
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before train.")
        return self.surrogate.fit(dataset, **kwargs)

    def predict(self, x_query: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before predict.")
        return self.surrogate.predict(np.asarray(x_query), **kwargs)

    def suggest_experiment(
        self,
        strategy: str,
        candidates: np.ndarray,
        budget: int = 1,
        costs: Optional[np.ndarray] = None,
    ) -> AcquisitionResult:
        strategy = strategy.lower()
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before suggest_experiment.")
        prediction = self.surrogate.predict(candidates, return_uncertainty=True)
        if prediction.std is None:
            scores = np.zeros(candidates.shape[0])
        else:
            scores = prediction.std.mean(axis=-1)
        if strategy == "bald":
            score_fn = lambda _: bald_acquisition(scores)
        elif strategy == "variance":
            score_fn = lambda _: variance_acquisition(scores)
        elif strategy == "expected_improvement":
            best = float(np.min(prediction.mean))
            score_fn = lambda _: expected_improvement(prediction.mean.squeeze(), prediction.std.squeeze(), best)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        result = acquisition_loop(candidates, score_fn, k=budget, scores=None)
        if costs is not None:
            indices, utility = cost_aware_selection(result.scores, costs, budget)
            return AcquisitionResult(scores=utility, selected_indices=indices)
        return result

    def assimilate(
        self,
        observations: np.ndarray,
        method: str = "enkf",
        ensemble_size: int = 10,
        observation_operator: Optional[Any] = None,
        observation_noise: float = 1e-3,
        inflation: float = 1.0,
    ) -> EnKFResult:
        method = method.lower()
        if method != "enkf":
            raise ValueError("Only EnKF assimilation is supported in this lightweight implementation.")
        observations = np.asarray(observations)
        state_dim = observations.shape[-1]
        ensemble = observations + np.random.normal(scale=np.sqrt(observation_noise), size=(ensemble_size, state_dim))
        enkf = EnKF(observation_operator=observation_operator)
        return enkf.update(ensemble, observations, observation_noise, inflation=inflation)

    def build_multifidelity(self) -> MultiFidelityTrainer:
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before building multi-fidelity trainer.")
        return MultiFidelityTrainer(surrogate=self.surrogate)
