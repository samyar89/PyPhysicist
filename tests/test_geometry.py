import numpy as np

from PyPhysicist.geometry import tensors


def test_as_tensor_dtype():
    data = [1, 2, 3]
    tensor = tensors.as_tensor(data, dtype=float)
    assert tensor.dtype == float
    assert np.allclose(tensor, np.array(data, dtype=float))


def test_metric_inverse_and_index_ops():
    metric = np.diag([2.0, 3.0])
    inverse_metric = tensors.metric_inverse(metric)
    vector = np.array([1.0, 2.0])
    covector = tensors.lower_index(vector, metric)
    raised = tensors.raise_index(covector, inverse_metric)
    assert np.allclose(inverse_metric, np.diag([0.5, 1.0 / 3.0]))
    assert np.allclose(covector, np.array([2.0, 6.0]))
    assert np.allclose(raised, vector)


def test_partial_derivative_linear_field():
    field = lambda coords: np.array([2 * coords[0] + 1.0, -coords[1]])
    coords = np.array([1.5, -0.5])
    deriv_x = tensors.partial_derivative(field, coords, index=0)
    deriv_y = tensors.partial_derivative(field, coords, index=1)
    assert np.allclose(deriv_x, np.array([2.0, 0.0]), atol=1e-6)
    assert np.allclose(deriv_y, np.array([0.0, -1.0]), atol=1e-6)


def test_metric_derivatives_diagonal():
    metric = lambda coords: np.array([[1.0 + coords[0], 0.0], [0.0, 1.0 + coords[1]]])
    coords = np.array([0.2, -0.1])
    derivs = tensors.metric_derivatives(metric, coords)
    expected_dx = np.array([[1.0, 0.0], [0.0, 0.0]])
    expected_dy = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert derivs.shape == (2, 2, 2)
    assert np.allclose(derivs[..., 0], expected_dx, atol=1e-6)
    assert np.allclose(derivs[..., 1], expected_dy, atol=1e-6)


def test_christoffel_flat_metric_zero():
    metric = np.eye(2)
    gamma = tensors.christoffel_symbols(metric, coords=np.array([0.0, 0.0]))
    assert np.allclose(gamma, np.zeros((2, 2, 2)))


def test_covariant_derivative_identity_vector_flat_metric():
    metric = np.eye(2)
    vector = lambda coords: np.array([coords[0], coords[1]])
    coords = np.array([1.0, -2.0])
    covariant = tensors.covariant_derivative_vector(vector, metric, coords)
    assert np.allclose(covariant, np.eye(2), atol=1e-6)


def test_curvature_flat_metric_zero():
    metric = np.eye(2)
    coords = np.array([0.1, 0.2])
    riemann = tensors.riemann_tensor(metric, coords)
    ricci = tensors.ricci_tensor(metric, coords)
    scalar = tensors.scalar_curvature(metric, coords)
    assert np.allclose(riemann, np.zeros((2, 2, 2, 2)), atol=1e-6)
    assert np.allclose(ricci, np.zeros((2, 2)), atol=1e-6)
    assert np.allclose(scalar, 0.0, atol=1e-6)
