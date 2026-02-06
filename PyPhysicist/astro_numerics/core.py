"""AstroNumerics: Barnes-Hut N-body simulations with adaptive time stepping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Body:
    """State container for an N-body particle."""

    mass: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time_step: float = 0.0

    def copy(self) -> "Body":
        """Return a deep-ish copy of the body state."""
        return Body(
            mass=float(self.mass),
            position=np.array(self.position, dtype=float),
            velocity=np.array(self.velocity, dtype=float),
            acceleration=np.array(self.acceleration, dtype=float),
            time_step=float(self.time_step),
        )


class BarnesHutNode:
    """Barnes-Hut tree node with multipole moments through octupole order."""

    def __init__(self, center: np.ndarray, half_size: float, indices: np.ndarray) -> None:
        self.center = center
        self.half_size = half_size
        self.indices = indices
        self.children: list[BarnesHutNode] = []
        self.mass = 0.0
        self.com = np.zeros(3)
        self.quadrupole = np.zeros((3, 3))
        self.octupole = np.zeros((3, 3, 3))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class BarnesHutTree:
    """Barnes-Hut octree with octupole expansion for force approximation."""

    def __init__(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        theta: float = 0.7,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.theta = theta
        self.root = self._build_tree()

    def _build_tree(self) -> BarnesHutNode:
        mins = self.positions.min(axis=0)
        maxs = self.positions.max(axis=0)
        center = 0.5 * (mins + maxs)
        half_size = 0.5 * (maxs - mins).max()
        if half_size == 0.0:
            half_size = 1.0
        indices = np.arange(self.positions.shape[0])
        root = BarnesHutNode(center=center, half_size=half_size, indices=indices)
        self._subdivide(root)
        return root

    def _subdivide(self, node: BarnesHutNode) -> None:
        positions = self.positions[node.indices]
        masses = self.masses[node.indices]
        node.mass = float(masses.sum())
        if node.mass > 0:
            node.com = (masses[:, None] * positions).sum(axis=0) / node.mass
        else:
            node.com = node.center.copy()

        relative = positions - node.com
        node.quadrupole = self._compute_quadrupole(relative, masses)
        node.octupole = self._compute_octupole(relative, masses)

        if len(node.indices) <= 1:
            return

        child_indices = [[] for _ in range(8)]
        offsets = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )

        for idx in node.indices:
            pos = self.positions[idx]
            mask = (pos >= node.center).astype(int)
            octant = mask[0] * 4 + mask[1] * 2 + mask[2]
            child_indices[octant].append(idx)

        for octant, inds in enumerate(child_indices):
            if not inds:
                continue
            offset = offsets[octant] * 0.5 * node.half_size
            child_center = node.center + offset
            child = BarnesHutNode(
                center=child_center,
                half_size=node.half_size * 0.5,
                indices=np.array(inds, dtype=int),
            )
            self._subdivide(child)
            node.children.append(child)

    @staticmethod
    def _compute_quadrupole(relative: np.ndarray, masses: np.ndarray) -> np.ndarray:
        quad = np.zeros((3, 3))
        for r, m in zip(relative, masses, strict=False):
            r2 = np.dot(r, r)
            quad += m * (3.0 * np.outer(r, r) - r2 * np.eye(3))
        return quad

    @staticmethod
    def _compute_octupole(relative: np.ndarray, masses: np.ndarray) -> np.ndarray:
        octupole = np.zeros((3, 3, 3))
        for r, m in zip(relative, masses, strict=False):
            r2 = np.dot(r, r)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        delta_jk = 1.0 if j == k else 0.0
                        delta_ik = 1.0 if i == k else 0.0
                        delta_ij = 1.0 if i == j else 0.0
                        octupole[i, j, k] += m * (
                            5.0 * r[i] * r[j] * r[k]
                            - r2 * (r[i] * delta_jk + r[j] * delta_ik + r[k] * delta_ij)
                        )
        return octupole

    def compute_accelerations(
        self,
        softening: float = 1e-3,
        gravitational_constant: float = 1.0,
    ) -> np.ndarray:
        accelerations = np.zeros_like(self.positions)
        for i, position in enumerate(self.positions):
            accelerations[i] = self._accumulate_force(
                self.root,
                position,
                i,
                softening,
                gravitational_constant,
            )
        return accelerations

    def _accumulate_force(
        self,
        node: BarnesHutNode,
        position: np.ndarray,
        index: int,
        softening: float,
        gravitational_constant: float,
    ) -> np.ndarray:
        if node.mass == 0.0:
            return np.zeros(3)

        displacement = node.com - position
        distance = np.linalg.norm(displacement) + softening
        if node.is_leaf() and len(node.indices) == 1 and node.indices[0] == index:
            return np.zeros(3)

        size = node.half_size * 2.0
        if node.is_leaf() or (size / distance) < self.theta:
            return self._multipole_acceleration(
                displacement,
                node.mass,
                node.quadrupole,
                node.octupole,
                gravitational_constant,
            )

        acceleration = np.zeros(3)
        for child in node.children:
            acceleration += self._accumulate_force(
                child,
                position,
                index,
                softening,
                gravitational_constant,
            )
        return acceleration

    @staticmethod
    def _multipole_acceleration(
        displacement: np.ndarray,
        mass: float,
        quadrupole: np.ndarray,
        octupole: np.ndarray,
        gravitational_constant: float,
    ) -> np.ndarray:
        r = np.linalg.norm(displacement)
        if r == 0.0:
            return np.zeros(3)
        r2 = r * r
        r3 = r2 * r
        r5 = r3 * r2
        r7 = r5 * r2
        monopole = -gravitational_constant * mass * displacement / r3
        quad_term = np.zeros(3)
        quad_contract = displacement @ quadrupole @ displacement
        quad_term = (
            -gravitational_constant
            * (5.0 * quad_contract * displacement / r7 - 2.0 * quadrupole @ displacement / r5)
            * 0.5
        )
        oct_term = np.zeros(3)
        oct_contract = np.einsum("ijk,i,j,k", octupole, displacement, displacement, displacement)
        oct_vector = np.einsum("ijk,j,k->i", octupole, displacement, displacement)
        oct_term = (
            -gravitational_constant
            * (7.0 * oct_contract * displacement / r7 - 3.0 * oct_vector / r5)
            / 6.0
        )
        return monopole + quad_term + oct_term


def adaptive_time_steps(
    accelerations: np.ndarray,
    previous_accelerations: np.ndarray | None,
    min_dt: float,
    max_dt: float,
    safety: float = 0.25,
) -> np.ndarray:
    """Compute per-body adaptive time steps using local acceleration variance."""
    norms = np.linalg.norm(accelerations, axis=1) + 1e-12
    dt = safety / np.sqrt(norms)
    if previous_accelerations is not None:
        delta = np.linalg.norm(accelerations - previous_accelerations, axis=1)
        variance = delta / (norms + 1e-12)
        dt /= 1.0 + variance
    return np.clip(dt, min_dt, max_dt)


class AstroNumericsSimulation:
    """N-body simulator with Barnes-Hut octupole expansion and adaptive time steps."""

    def __init__(
        self,
        bodies: Iterable[Body],
        gravitational_constant: float = 1.0,
        theta: float = 0.7,
        softening: float = 1e-3,
        min_dt: float = 1e-4,
        max_dt: float = 1e-1,
    ) -> None:
        self.bodies = [body.copy() for body in bodies]
        self.gravitational_constant = gravitational_constant
        self.theta = theta
        self.softening = softening
        self.min_dt = min_dt
        self.max_dt = max_dt
        self._previous_accelerations: np.ndarray | None = None

    def step(self) -> None:
        positions = np.array([body.position for body in self.bodies])
        masses = np.array([body.mass for body in self.bodies])
        tree = BarnesHutTree(positions, masses, theta=self.theta)
        accelerations = tree.compute_accelerations(
            softening=self.softening,
            gravitational_constant=self.gravitational_constant,
        )
        time_steps = adaptive_time_steps(
            accelerations,
            self._previous_accelerations,
            min_dt=self.min_dt,
            max_dt=self.max_dt,
        )
        for body, acceleration, dt in zip(self.bodies, accelerations, time_steps, strict=False):
            body.velocity = body.velocity + acceleration * dt
            body.position = body.position + body.velocity * dt
            body.acceleration = acceleration
            body.time_step = dt
        self._previous_accelerations = accelerations

    def simulate(self, steps: int) -> np.ndarray:
        """Return trajectory array with shape (steps, bodies, 3)."""
        trajectory = np.zeros((steps, len(self.bodies), 3))
        for step in range(steps):
            trajectory[step] = np.array([body.position for body in self.bodies])
            self.step()
        return trajectory


def keplerian_orbital_elements(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float,
) -> dict[str, float]:
    """Compute Keplerian orbital elements from Cartesian state vectors."""
    r = np.array(position, dtype=float)
    v = np.array(velocity, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    n = np.cross([0.0, 0.0, 1.0], h)
    n_norm = np.linalg.norm(n)
    e_vec = (np.cross(v, h) / mu) - r / r_norm
    e = np.linalg.norm(e_vec)
    energy = 0.5 * v_norm**2 - mu / r_norm
    a = -mu / (2.0 * energy) if energy != 0 else np.inf
    i = np.arccos(h[2] / h_norm) if h_norm != 0 else 0.0
    raan = np.arccos(n[0] / n_norm) if n_norm != 0 else 0.0
    if n_norm != 0 and n[1] < 0:
        raan = 2.0 * np.pi - raan
    argp = np.arccos(np.dot(n, e_vec) / (n_norm * e)) if n_norm != 0 and e != 0 else 0.0
    if e_vec[2] < 0 and n_norm != 0 and e != 0:
        argp = 2.0 * np.pi - argp
    true_anomaly = np.arccos(np.dot(e_vec, r) / (e * r_norm)) if e != 0 else 0.0
    if np.dot(r, v) < 0:
        true_anomaly = 2.0 * np.pi - true_anomaly
    return {
        "semi_major_axis": float(a),
        "eccentricity": float(e),
        "inclination": float(i),
        "raan": float(raan),
        "argument_of_periapsis": float(argp),
        "true_anomaly": float(true_anomaly),
    }


def lyapunov_exponent(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    dt: float,
    renormalization_interval: int = 10,
) -> float:
    """Estimate the long-term Lyapunov exponent from two nearby trajectories."""
    separation = trajectory_b - trajectory_a
    distances = np.linalg.norm(separation.reshape(separation.shape[0], -1), axis=1)
    if distances[0] == 0:
        return 0.0
    log_sum = 0.0
    count = 0
    for idx in range(renormalization_interval, len(distances), renormalization_interval):
        if distances[idx] == 0:
            continue
        log_sum += np.log(distances[idx] / distances[0])
        count += 1
    if count == 0:
        return 0.0
    total_time = count * renormalization_interval * dt
    return log_sum / total_time


def lagrange_points(m1: float, m2: float, distance: float) -> dict[str, np.ndarray]:
    """Compute L1-L5 points for a three-body system in a rotating frame."""
    total_mass = m1 + m2
    mu = m2 / total_mass
    x1 = -mu * distance
    x2 = (1.0 - mu) * distance

    def colinear_equation(x: float) -> float:
        r1 = x - x1
        r2 = x - x2
        return x - (1.0 - mu) * r1 / abs(r1) ** 3 - mu * r2 / abs(r2) ** 3

    def solve_interval(a: float, b: float, iterations: int = 100) -> float:
        fa = colinear_equation(a)
        fb = colinear_equation(b)
        if fa == 0:
            return a
        if fb == 0:
            return b
        for _ in range(iterations):
            mid = 0.5 * (a + b)
            fm = colinear_equation(mid)
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        return 0.5 * (a + b)

    l1 = solve_interval(x2 - 0.7 * distance, x2 - 1e-6)
    l2 = solve_interval(x2 + 1e-6, x2 + 0.7 * distance)
    l3 = solve_interval(x1 - 0.7 * distance, x1 - 1e-6)
    l4 = np.array([0.5 * (x1 + x2), np.sqrt(3) * 0.5 * distance, 0.0])
    l5 = np.array([0.5 * (x1 + x2), -np.sqrt(3) * 0.5 * distance, 0.0])
    return {
        "L1": np.array([l1, 0.0, 0.0]),
        "L2": np.array([l2, 0.0, 0.0]),
        "L3": np.array([l3, 0.0, 0.0]),
        "L4": l4,
        "L5": l5,
    }
