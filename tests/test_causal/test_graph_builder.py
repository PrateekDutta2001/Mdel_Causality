"""Tests for causal graph construction."""

import networkx as nx

from app.causal.graph_builder import DEFAULT_EDGES, get_causal_graph


def test_causal_graph_is_dag() -> None:
    graph = get_causal_graph()
    assert isinstance(graph, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(graph)


def test_causal_graph_contains_expected_edges() -> None:
    graph = get_causal_graph()
    for src, dst in DEFAULT_EDGES:
        assert graph.has_edge(src, dst)
