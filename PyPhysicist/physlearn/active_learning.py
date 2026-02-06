"""Active learning acquisition functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class AcquisitionResult:
    scores: np.ndarray
    selected_indices: np.ndarray


def bald_acquisition(variances: np.ndarray) -> np.ndarray:
    """BALD-style acquisition using predictive variance as proxy."""

    variances = np.asarray(variances)
    return variances


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    mean = np.asarray(mean)
    std = np.asarray(std)
    std_safe = np.where(std == 0, 1e-12, std)
    z = (best - mean) / std_safe
    cdf = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    return (best - mean) * cdf + std_safe * pdf


def variance_reduction(variances: np.ndarray) -> np.ndarray:
    return np.asarray(variances)


def query_by_committee(predictions: np.ndarray) -> np.ndarray:
    """Variance across committee members."""

    predictions = np.asarray(predictions)
    return np.var(predictions, axis=0)


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")
    scores = np.asarray(scores)
    return np.argsort(scores)[-k:][::-1]


def acquisition_loop(
    candidates: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    k: int = 1,
    scores: Optional[np.ndarray] = None,
) -> AcquisitionResult:
    if scores is None:
        scores = score_fn(candidates)
    indices = select_top_k(scores, k)
    return AcquisitionResult(scores=scores, selected_indices=indices)
