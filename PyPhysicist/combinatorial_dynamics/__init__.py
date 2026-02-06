"""Combinatorial dynamics submodule for PyPhysicist."""

from .core import (
    CombinatorialEnergyResult,
    GraphCellularAutomaton,
    StackEvolutionModel,
    StackEvolutionResult,
    catalan_number,
    combinatorial_energy_minimization,
    dyck_word_from_actions,
    graph_diffusion,
    graph_spectral_metrics,
    scale_free_graph,
    simulate_gca,
    small_world_graph,
)

__all__ = [
    "CombinatorialEnergyResult",
    "GraphCellularAutomaton",
    "StackEvolutionModel",
    "StackEvolutionResult",
    "catalan_number",
    "combinatorial_energy_minimization",
    "dyck_word_from_actions",
    "graph_diffusion",
    "graph_spectral_metrics",
    "scale_free_graph",
    "simulate_gca",
    "small_world_graph",
]
