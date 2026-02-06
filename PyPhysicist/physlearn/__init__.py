"""PhysLearn: physics-aware hybrid surrogate & data assimilation suite."""

from .core import PhysLearn
from .registry import ModelRegistry, SolverRegistry
from .metrics import calibration_error, coverage_probability, crps_gaussian, nll_gaussian, rmse, nrmse
from .active_learning import (
    AcquisitionResult,
    bald_acquisition,
    expected_improvement,
    query_by_committee,
    variance_reduction,
)
from .config import ExperimentConfig
from .data_assimilation import EnKF, EnKFResult
from .models.base import BaseSurrogate, SurrogatePrediction

__all__ = [
    "PhysLearn",
    "ModelRegistry",
    "SolverRegistry",
    "BaseSurrogate",
    "SurrogatePrediction",
    "EnKF",
    "EnKFResult",
    "AcquisitionResult",
    "bald_acquisition",
    "expected_improvement",
    "query_by_committee",
    "variance_reduction",
    "ExperimentConfig",
    "rmse",
    "nrmse",
    "nll_gaussian",
    "crps_gaussian",
    "coverage_probability",
    "calibration_error",
]
