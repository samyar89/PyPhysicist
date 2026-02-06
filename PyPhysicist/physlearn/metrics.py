"""Evaluation metrics for PhysLearn."""

from __future__ import annotations

import numpy as np


def _erf_inv(x: np.ndarray) -> np.ndarray:
    """Approximate inverse error function using Winitzki approximation."""
    x = np.asarray(x)
    a = 0.147  # approximation constant
    sign = np.sign(x)
    ln = np.log(1 - x**2)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sign * np.sqrt(np.sqrt(first**2 - second) - first)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    denom = np.mean(np.abs(target))
    if denom == 0:
        return float(rmse(pred, target))
    return float(rmse(pred, target) / denom)


def nll_gaussian(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> float:
    mean = np.asarray(mean)
    std = np.asarray(std)
    target = np.asarray(target)
    var = np.clip(std**2, 1e-12, None)
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + ((target - mean) ** 2) / var))


def crps_gaussian(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> float:
    mean = np.asarray(mean)
    std = np.asarray(std)
    target = np.asarray(target)
    z = (target - mean) / np.clip(std, 1e-12, None)
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    cdf = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    score = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(score))


def coverage_probability(mean: np.ndarray, std: np.ndarray, target: np.ndarray, alpha: float = 0.95) -> float:
    mean = np.asarray(mean)
    std = np.asarray(std)
    target = np.asarray(target)
    z = np.sqrt(2.0) * _erf_inv(alpha)
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((target >= lower) & (target <= upper)))


def calibration_error(mean: np.ndarray, std: np.ndarray, target: np.ndarray, alpha: float = 0.95) -> float:
    coverage = coverage_probability(mean, std, target, alpha=alpha)
    return float(abs(coverage - alpha))
