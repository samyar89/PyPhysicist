"""Core PhysLearn API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .active_learning import AcquisitionResult, acquisition_loop, bald_acquisition, variance_reduction
from .data_assimilation import EnKF, EnKFResult
from .models.base import BaseSurrogate, SurrogatePrediction
from .models.simple import EnsembleSurrogate, IdentitySurrogate, LinearSurrogate, PlaceholderSurrogate
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
            "fno": lambda **kwargs: PlaceholderSurrogate(name="FNO", config=kwargs),
            "deeponet": lambda **kwargs: PlaceholderSurrogate(name="DeepONet", config=kwargs),
            "pinn": lambda **kwargs: PlaceholderSurrogate(name="PINN", config=kwargs),
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
        x = dataset.get("x")
        y = dataset.get("y")
        if x is None or y is None:
            raise ValueError("Dataset must contain 'x' and 'y' arrays.")
        return self.surrogate.fit(np.asarray(x), np.asarray(y), **kwargs)

    def predict(self, x_query: np.ndarray, **kwargs: Any) -> SurrogatePrediction:
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before predict.")
        return self.surrogate.predict(np.asarray(x_query), **kwargs)

    def suggest_experiment(
        self,
        strategy: str,
        candidates: np.ndarray,
        budget: int = 1,
    ) -> AcquisitionResult:
        strategy = strategy.lower()
        if self.surrogate is None:
            raise RuntimeError("Call build_surrogate before suggest_experiment.")
        prediction = self.surrogate.predict(candidates)
        if prediction.std is None:
            scores = np.zeros(candidates.shape[0])
        else:
            scores = prediction.std.mean(axis=-1)
        if strategy == "bald":
            score_fn = lambda _: bald_acquisition(scores)
        elif strategy == "variance_reduction":
            score_fn = lambda _: variance_reduction(scores)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")
        return acquisition_loop(candidates, score_fn, k=budget, scores=scores)

    def assimilate(
        self,
        observations: np.ndarray,
        method: str = "enkf",
        ensemble_size: int = 10,
        observation_operator: Optional[Any] = None,
        observation_noise: float = 1e-3,
    ) -> EnKFResult:
        method = method.lower()
        if method != "enkf":
            raise ValueError("Only EnKF assimilation is supported in this lightweight implementation.")
        observations = np.asarray(observations)
        state_dim = observations.shape[-1]
        ensemble = observations + np.random.normal(scale=np.sqrt(observation_noise), size=(ensemble_size, state_dim))
        enkf = EnKF(observation_operator=observation_operator)
        return enkf.update(ensemble, observations, observation_noise)
