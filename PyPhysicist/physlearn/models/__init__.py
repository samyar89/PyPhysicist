"""Model components for PhysLearn."""

from .base import BaseSurrogate, SurrogatePrediction
from .simple import EnsembleSurrogate, IdentitySurrogate, LinearSurrogate, PlaceholderSurrogate

__all__ = [
    "BaseSurrogate",
    "SurrogatePrediction",
    "LinearSurrogate",
    "EnsembleSurrogate",
    "IdentitySurrogate",
    "PlaceholderSurrogate",
]
