"""Data assimilation tools such as Ensemble Kalman Filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class EnKFResult:
    analysis_ensemble: np.ndarray
    analysis_mean: np.ndarray
    analysis_cov: np.ndarray


class EnKF:
    """Simple Ensemble Kalman Filter implementation."""

    def __init__(self, observation_operator: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.observation_operator = observation_operator or (lambda x: x)

    def update(
        self,
        ensemble: np.ndarray,
        observations: np.ndarray,
        observation_noise: float,
        inflation: float = 1.0,
    ) -> EnKFResult:
        ensemble = np.asarray(ensemble)
        observations = np.asarray(observations)
        if ensemble.ndim != 2:
            raise ValueError("Ensemble must be 2D (n_members, state_dim).")
        if inflation <= 0:
            raise ValueError("Inflation factor must be positive.")
        n_members = ensemble.shape[0]
        state_mean = ensemble.mean(axis=0)
        ensemble = state_mean + inflation * (ensemble - state_mean)
        obs = self.observation_operator(ensemble)
        obs_mean = obs.mean(axis=0)
        obs_anom = obs - obs_mean
        state_anom = ensemble - ensemble.mean(axis=0)
        obs_cov = (obs_anom.T @ obs_anom) / (n_members - 1)
        cross_cov = (state_anom.T @ obs_anom) / (n_members - 1)
        obs_cov += np.eye(obs_cov.shape[0]) * observation_noise
        kalman_gain = cross_cov @ np.linalg.inv(obs_cov)
        innovations = observations - obs
        analysis = ensemble + innovations @ kalman_gain.T
        analysis_mean = analysis.mean(axis=0)
        analysis_cov = np.cov(analysis, rowvar=False)
        return EnKFResult(
            analysis_ensemble=analysis,
            analysis_mean=analysis_mean,
            analysis_cov=analysis_cov,
        )
