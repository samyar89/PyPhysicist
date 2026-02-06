"""Model components for PhysLearn."""

from .base import BaseSurrogate, SurrogatePrediction
from .deeponet import DeepONetSurrogate
from .fno import FNOOperatorSurrogate
from .hybrid import NeuralCorrectorSurrogate
from .pinn import PINNSurrogate
from .simple import EnsembleSurrogate, IdentitySurrogate, LinearSurrogate, PlaceholderSurrogate

__all__ = [
    "BaseSurrogate",
    "SurrogatePrediction",
    "LinearSurrogate",
    "EnsembleSurrogate",
    "IdentitySurrogate",
    "PlaceholderSurrogate",
    "FNOOperatorSurrogate",
    "DeepONetSurrogate",
    "PINNSurrogate",
    "NeuralCorrectorSurrogate",
]
