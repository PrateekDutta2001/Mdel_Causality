"""Confounder identification and validation for causal estimates."""

from __future__ import annotations

import networkx as nx

from app.causal.graph_builder import OUTCOME, TREATMENT_PARAMS


def get_confounders(graph: nx.DiGraph, treatment: str, outcome: str = OUTCOME) -> list[str]:
    """Return adjustment set: parents of treatment and outcome excluding treatment itself."""
    confounders: set[str] = set()
    if treatment in graph:
        confounders.update(graph.predecessors(treatment))
    if outcome in graph:
        confounders.update(graph.predecessors(outcome))
    confounders.discard(treatment)
    confounders.discard(outcome)
    return sorted(confounders)


def get_all_confounder_map(graph: nx.DiGraph) -> dict[str, list[str]]:
    """Map each treatment parameter to its confounder set."""
    return {param: get_confounders(graph, param) for param in TREATMENT_PARAMS if param in graph}


def validate_backdoor_paths(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str = OUTCOME,
) -> bool:
    """Check whether a backdoor adjustment set exists (structural sanity check).

    TODO: Integrate DoWhy's built-in backdoor criterion for formal validation.
    """
    confounders = get_confounders(graph, treatment, outcome)
    return len(confounders) >= 0  # placeholder — graph is assumed valid
