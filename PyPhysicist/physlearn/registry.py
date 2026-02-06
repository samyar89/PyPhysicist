"""Registries for solvers and surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class SolverRegistry:
    """Registry for physics solvers keyed by fidelity or name."""

    _solvers: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, solver: Callable[..., Any]) -> None:
        if not callable(solver):
            raise TypeError("Solver must be callable.")
        self._solvers[name] = solver

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._solvers:
            raise KeyError(f"Solver '{name}' not registered.")
        return self._solvers[name]

    def list(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._solvers)


@dataclass
class ModelRegistry:
    """Registry for surrogate model builders."""

    _builders: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        if not callable(builder):
            raise TypeError("Model builder must be callable.")
        self._builders[name.lower()] = builder

    def build(self, name: str, **kwargs: Any) -> Any:
        key = name.lower()
        if key not in self._builders:
            raise KeyError(f"Model '{name}' not registered.")
        return self._builders[key](**kwargs)

    def list(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._builders)

    def ensure_default_models(self, default_builders: Optional[Dict[str, Callable[..., Any]]] = None) -> None:
        if default_builders is None:
            return
        for name, builder in default_builders.items():
            if name.lower() not in self._builders:
                self._builders[name.lower()] = builder
