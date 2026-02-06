"""PhysLearn: physics-aware hybrid surrogate & data assimilation suite."""

from .core import PhysLearn
from .registry import ModelRegistry, SolverRegistry
from .metrics import calibration_error, coverage_probability, crps_gaussian, nll_gaussian, rmse, nrmse
from .active_learning import (
    AcquisitionResult,
    bald_acquisition,
    cost_aware_selection,
    expected_improvement,
    query_by_committee,
    variance_acquisition,
    variance_reduction,
)
from .config import ExperimentConfig
from .data_assimilation import EnKF, EnKFResult
from .models.base import BaseSurrogate, SurrogatePrediction
from .multifidelity import MultiFidelityTrainer, discrepancy_dataset
from .datasets import burgers_1d, heat_1d, low_high_fidelity_pair, shallow_water_toy
from .train import TrainConfig
from .hooks import EarlyStopping, History
from .uq import EnsembleUQ, gpytorch_available, mc_dropout_predict

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
    "cost_aware_selection",
    "expected_improvement",
    "query_by_committee",
    "variance_acquisition",
    "variance_reduction",
    "ExperimentConfig",
    "rmse",
    "nrmse",
    "nll_gaussian",
    "crps_gaussian",
    "coverage_probability",
    "calibration_error",
    "MultiFidelityTrainer",
    "discrepancy_dataset",
    "burgers_1d",
    "heat_1d",
    "shallow_water_toy",
    "low_high_fidelity_pair",
    "TrainConfig",
    "EarlyStopping",
    "History",
    "EnsembleUQ",
    "mc_dropout_predict",
    "gpytorch_available",
]
