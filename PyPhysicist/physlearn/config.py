"""Configuration utilities for PhysLearn experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    name: str = "physlearn-run"
    seed: int = 42
    device: str = "cpu"
    model: str = "linear"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    uq: Dict[str, Any] = field(default_factory=dict)
    tracking: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
