"""Combinatorial dynamics tools linking discrete math and physical dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

StateVector = np.ndarray
RuleFunction = Callable[[int, float, Sequence[float], nx.Graph], float]


def scale_free_graph(
    node_count: int,
    attachment_edges: int = 2,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a scale-free (Barabási-Albert) graph."""
    return nx.barabasi_albert_graph(node_count, attachment_edges, seed=seed)


def small_world_graph(
    node_count: int,
    nearest_neighbors: int,
    rewiring_prob: float,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Generate a small-world (Watts-Strogatz) graph."""
    return nx.watts_strogatz_graph(
        node_count,
        nearest_neighbors,
        rewiring_prob,
        seed=seed,
    )


def _to_networkx_graph(graph: object) -> nx.Graph:
    if isinstance(graph, nx.Graph):
        return graph
    if isinstance(graph, np.ndarray):
        return nx.from_numpy_array(graph)
    if isinstance(graph, Mapping):
        return nx.from_dict_of_lists(graph)
    if isinstance(graph, Iterable):
        return nx.from_edgelist(graph)
    raise TypeError("Unsupported graph representation for combinatorial dynamics.")


def _state_vector(
    graph: nx.Graph,
    state: Optional[Mapping[int, float] | Sequence[float]],
) -> Tuple[StateVector, List[int]]:
    nodes = list(graph.nodes())
    if state is None:
        return np.zeros(len(nodes), dtype=float), nodes
    if isinstance(state, Mapping):
        vector = np.array([state.get(node, 0.0) for node in nodes], dtype=float)
        return vector, nodes
    vector = np.asarray(state, dtype=float)
    if vector.shape[0] != len(nodes):
        raise ValueError("State vector length must match number of graph nodes.")
    return vector, nodes


class GraphCellularAutomaton:
    """Graph cellular automaton on arbitrary topology."""

    def __init__(
        self,
        graph: object,
        rule: RuleFunction,
        state: Optional[Mapping[int, float] | Sequence[float]] = None,
    ) -> None:
        self.graph = _to_networkx_graph(graph)
        self.rule = rule
        self.state, self._nodes = _state_vector(self.graph, state)

    def step(self) -> StateVector:
        """Advance the automaton by one step."""
        new_state = np.zeros_like(self.state, dtype=float)
        node_index = {node: idx for idx, node in enumerate(self._nodes)}
        for idx, node in enumerate(self._nodes):
            neighbors = [node_index[nbr] for nbr in self.graph.neighbors(node)]
            neighbor_states = self.state[neighbors] if neighbors else np.array([])
            new_state[idx] = self.rule(
                node,
                float(self.state[idx]),
                neighbor_states,
                self.graph,
            )
        self.state = new_state
        return self.state

    def run(self, steps: int) -> np.ndarray:
        """Run the automaton for multiple steps and return the trajectory."""
        trajectory = np.zeros((steps + 1, len(self.state)), dtype=float)
        trajectory[0] = self.state
        for step in range(steps):
            trajectory[step + 1] = self.step()
        return trajectory


def simulate_gca(
    graph: object,
    rule: RuleFunction,
    steps: int,
    state: Optional[Mapping[int, float] | Sequence[float]] = None,
) -> np.ndarray:
    """Convenience wrapper to simulate a graph cellular automaton."""
    automaton = GraphCellularAutomaton(graph, rule, state=state)
    return automaton.run(steps)


def catalan_number(n: int) -> int:
    """Compute the nth Catalan number."""
    if n < 0:
        raise ValueError("Catalan numbers are defined for n >= 0.")
    return math.comb(2 * n, n) // (n + 1)


@dataclass(frozen=True)
class StackEvolutionResult:
    """Result from a stack-based Catalan evolution simulation."""

    depths: Tuple[int, ...]
    actions: Tuple[str, ...]
    stability_index: float
    max_depth: int
    mean_depth: float


class StackEvolutionModel:
    """Stack-based evolution that respects Catalan (Dyck path) constraints."""

    def __init__(
        self,
        push_bias: float = 0.5,
        max_depth: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= push_bias <= 1.0:
            raise ValueError("push_bias must be between 0 and 1.")
        self.push_bias = push_bias
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed)

    def step(self, depth: int) -> Tuple[int, str]:
        """Perform one Catalan-compliant stack step."""
        can_push = self.max_depth is None or depth < self.max_depth
        can_pop = depth > 0
        if not can_pop and not can_push:
            return depth, "hold"
        if not can_pop:
            return depth + 1, "push"
        if not can_push:
            return depth - 1, "pop"
        action = "push" if self.rng.random() < self.push_bias else "pop"
        return (depth + 1, "push") if action == "push" else (depth - 1, "pop")

    def simulate(self, steps: int, start_depth: int = 0) -> StackEvolutionResult:
        """Simulate stack evolution for a number of steps."""
        depth = start_depth
        depths: List[int] = [depth]
        actions: List[str] = []
        for _ in range(steps):
            depth, action = self.step(depth)
            depths.append(depth)
            actions.append(action)
        depth_array = np.array(depths, dtype=float)
        stability_index = float(1.0 / (1.0 + np.var(depth_array)))
        return StackEvolutionResult(
            depths=tuple(depths),
            actions=tuple(actions),
            stability_index=stability_index,
            max_depth=int(np.max(depth_array)),
            mean_depth=float(np.mean(depth_array)),
        )


def dyck_word_from_actions(actions: Sequence[str]) -> str:
    """Translate push/pop actions into a Dyck word."""
    mapping = {"push": "(", "pop": ")", "hold": ""}
    return "".join(mapping.get(action, "") for action in actions)


def _adjacency_matrix(graph: nx.Graph, nodes: Sequence[int]) -> np.ndarray:
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    return adjacency


def graph_spectral_metrics(
    graph: object,
    state: Optional[Sequence[float] | Mapping[int, float]] = None,
    use_laplacian: bool = True,
) -> dict:
    """Map physical state evolution onto spectral graph metrics."""
    nx_graph = _to_networkx_graph(graph)
    vector, nodes = _state_vector(nx_graph, state)
    adjacency = _adjacency_matrix(nx_graph, nodes)
    degrees = np.sum(adjacency, axis=1)
    laplacian = np.diag(degrees) - adjacency
    operator = laplacian if use_laplacian else adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if eigenvalues.size > 1 else 0.0
    state_energy = float(vector @ operator @ vector)
    projection = eigenvectors.T @ vector
    high_frequency_energy = float(np.sum(projection[eigenvalues > np.median(eigenvalues)] ** 2))
    return {
        "spectral_radius": spectral_radius,
        "spectral_gap": spectral_gap,
        "state_energy": state_energy,
        "high_frequency_energy": high_frequency_energy,
        "eigenvalues": eigenvalues,
    }


def graph_diffusion(
    graph: object,
    initial_state: Sequence[float] | Mapping[int, float],
    steps: int,
    diffusion_rate: float = 0.1,
    normalized: bool = True,
) -> np.ndarray:
    """Diffuse a state across a graph using the Laplacian operator."""
    nx_graph = _to_networkx_graph(graph)
    state, nodes = _state_vector(nx_graph, initial_state)
    adjacency = _adjacency_matrix(nx_graph, nodes)
    degrees = np.sum(adjacency, axis=1)
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sqrt_deg = np.diag(1.0 / np.sqrt(np.where(degrees == 0, 1.0, degrees)))
        laplacian = np.eye(len(nodes)) - inv_sqrt_deg @ adjacency @ inv_sqrt_deg
    else:
        laplacian = np.diag(degrees) - adjacency
    trajectory = np.zeros((steps + 1, len(state)), dtype=float)
    trajectory[0] = state
    current = state.copy()
    for step in range(steps):
        current = current - diffusion_rate * (laplacian @ current)
        trajectory[step + 1] = current
    return trajectory


@dataclass(frozen=True)
class CombinatorialEnergyResult:
    """Result from combinatorial energy minimization."""

    best_state: object
    best_energy: float
    energies: Tuple[float, ...]


def combinatorial_energy_minimization(
    initial_state: object,
    neighbor_fn: Callable[[object], Iterable[object]],
    energy_fn: Callable[[object], float],
    steps: int = 200,
    temperature: float = 1.0,
    cooling_rate: float = 0.99,
    seed: Optional[int] = None,
) -> CombinatorialEnergyResult:
    """Simulated annealing over combinatorial state spaces."""
    rng = np.random.default_rng(seed)
    current_state = initial_state
    current_energy = energy_fn(current_state)
    best_state = current_state
    best_energy = current_energy
    energies: List[float] = [current_energy]
    temp = temperature
    for _ in range(steps):
        neighbors = list(neighbor_fn(current_state))
        if not neighbors:
            energies.append(current_energy)
            temp *= cooling_rate
            continue
        candidate = neighbors[int(rng.integers(0, len(neighbors)))]
        candidate_energy = energy_fn(candidate)
        accept = candidate_energy < current_energy
        if not accept:
            delta = candidate_energy - current_energy
            accept = rng.random() < math.exp(-delta / max(temp, 1e-8))
        if accept:
            current_state = candidate
            current_energy = candidate_energy
        if current_energy < best_energy:
            best_energy = current_energy
            best_state = current_state
        energies.append(current_energy)
        temp *= cooling_rate
    return CombinatorialEnergyResult(
        best_state=best_state,
        best_energy=float(best_energy),
        energies=tuple(energies),
    )
