"""Tensor manipulation and differential geometry utilities."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

import numpy as np

ArrayLike = Union[float, Iterable[float], np.ndarray]
MetricInput = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
TensorField = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def as_tensor(data: ArrayLike, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert data into a NumPy array with an optional dtype."""
    return np.asarray(data, dtype=dtype)


def metric_inverse(metric: np.ndarray) -> np.ndarray:
    """Return the inverse of a metric tensor."""
    metric = np.asarray(metric)
    return np.linalg.inv(metric)


def lower_index(vector: ArrayLike, metric: np.ndarray) -> np.ndarray:
    """Lower a vector index using the metric: v_i = g_{ij} v^j."""
    vector = np.asarray(vector)
    metric = np.asarray(metric)
    return np.einsum("ij,...j->...i", metric, vector)


def raise_index(covector: ArrayLike, inverse_metric: np.ndarray) -> np.ndarray:
    """Raise a covector index using the inverse metric: v^i = g^{ij} v_j."""
    covector = np.asarray(covector)
    inverse_metric = np.asarray(inverse_metric)
    return np.einsum("ij,...j->...i", inverse_metric, covector)


def _evaluate_field(field: TensorField, coords: np.ndarray) -> np.ndarray:
    """Evaluate a tensor field or return a constant array."""
    if callable(field):
        return np.asarray(field(coords))
    return np.asarray(field)


def partial_derivative(
    field: TensorField,
    coords: ArrayLike,
    index: int,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute a numerical partial derivative of a tensor field.

    Uses a central difference scheme for numerical stability.
    """
    coords = np.asarray(coords, dtype=float)
    shift = np.zeros_like(coords)
    shift[index] = step
    forward = _evaluate_field(field, coords + shift)
    backward = _evaluate_field(field, coords - shift)
    return (forward - backward) / (2 * step)


def metric_derivatives(
    metric: MetricInput,
    coords: ArrayLike,
    step: float = 1e-5,
) -> np.ndarray:
    """Return partial derivatives of the metric tensor.

    Output shape is (dim, dim, dim) with the last axis indicating
    differentiation with respect to a coordinate index.
    """
    coords = np.asarray(coords, dtype=float)
    base_metric = _evaluate_field(metric, coords)
    dim = base_metric.shape[0]
    derivatives = np.zeros((dim, dim, dim), dtype=base_metric.dtype)
    for index in range(dim):
        derivatives[..., index] = partial_derivative(metric, coords, index, step=step)
    return derivatives


def christoffel_symbols(
    metric: MetricInput,
    coords: ArrayLike,
    metric_derivs: Optional[np.ndarray] = None,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute Christoffel symbols of the second kind.

    Gamma^i_{jk} = 1/2 g^{il}(∂_j g_{kl} + ∂_k g_{jl} - ∂_l g_{jk}).
    """
    coords = np.asarray(coords, dtype=float)
    metric_value = _evaluate_field(metric, coords)
    inverse_metric = metric_inverse(metric_value)
    if metric_derivs is None:
        metric_derivs = metric_derivatives(metric, coords, step=step)
    dim = metric_value.shape[0]
    gamma = np.zeros((dim, dim, dim), dtype=metric_value.dtype)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                term = 0.0
                for l in range(dim):
                    term += inverse_metric[i, l] * (
                        metric_derivs[k, l, j]
                        + metric_derivs[j, l, k]
                        - metric_derivs[j, k, l]
                    )
                gamma[i, j, k] = 0.5 * term
    return gamma


def covariant_derivative_vector(
    vector: TensorField,
    metric: MetricInput,
    coords: ArrayLike,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute the covariant derivative of a contravariant vector field.

    Returns ∇_j v^i with shape (dim, dim).
    """
    coords = np.asarray(coords, dtype=float)
    vector_value = _evaluate_field(vector, coords)
    dim = vector_value.shape[0]
    gamma = christoffel_symbols(metric, coords, step=step)
    covariant = np.zeros((dim, dim), dtype=vector_value.dtype)
    for j in range(dim):
        partial = partial_derivative(vector, coords, j, step=step)
        covariant[:, j] = partial
        for i in range(dim):
            covariant[i, j] += np.sum(gamma[i, j, :] * vector_value)
    return covariant


def riemann_tensor(
    metric: MetricInput,
    coords: ArrayLike,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute the Riemann curvature tensor R^i_{jkl}."""
    coords = np.asarray(coords, dtype=float)
    gamma = christoffel_symbols(metric, coords, step=step)
    dim = gamma.shape[0]
    riemann = np.zeros((dim, dim, dim, dim), dtype=gamma.dtype)
    for k in range(dim):
        for l in range(dim):
            gamma_k = partial_derivative(lambda x: christoffel_symbols(metric, x, step=step), coords, k, step=step)
            gamma_l = partial_derivative(lambda x: christoffel_symbols(metric, x, step=step), coords, l, step=step)
            for i in range(dim):
                for j in range(dim):
                    term = gamma_k[i, j, l] - gamma_l[i, j, k]
                    term += np.sum(gamma[i, k, :] * gamma[:, j, l])
                    term -= np.sum(gamma[i, l, :] * gamma[:, j, k])
                    riemann[i, j, k, l] = term
    return riemann


def ricci_tensor(
    metric: MetricInput,
    coords: ArrayLike,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute the Ricci tensor by contracting the Riemann tensor."""
    riemann = riemann_tensor(metric, coords, step=step)
    return np.einsum("iikl->kl", riemann)


def scalar_curvature(
    metric: MetricInput,
    coords: ArrayLike,
    step: float = 1e-5,
) -> np.ndarray:
    """Compute the scalar curvature R = g^{ij} R_{ij}."""
    coords = np.asarray(coords, dtype=float)
    metric_value = _evaluate_field(metric, coords)
    inverse_metric = metric_inverse(metric_value)
    ricci = ricci_tensor(metric, coords, step=step)
    return np.einsum("ij,ij->", inverse_metric, ricci)


__all__ = [
    "ArrayLike",
    "MetricInput",
    "TensorField",
    "as_tensor",
    "metric_inverse",
    "lower_index",
    "raise_index",
    "partial_derivative",
    "metric_derivatives",
    "christoffel_symbols",
    "covariant_derivative_vector",
    "riemann_tensor",
    "ricci_tensor",
    "scalar_curvature",
]
