import pytest
import networkx as nx
from splendor.loops import LoopData, get_loops

def _graph_with_attrs(G: nx.Graph, length_m: float = 1.) -> nx.Graph:
    """Add get_loops' expected edge_id and length_m attributes."""
    for eid, (u, v) in enumerate(G.edges()):
        G[u][v]["edge_id"] = eid
        G[u][v]["length_m"] = length_m
    return G


@pytest.fixture
def k3_graph() -> nx.Graph:
    """k3 with unit lengths."""
    return _graph_with_attrs(nx.complete_graph(3))


@pytest.fixture
def k3_loops(k3_graph) -> LoopData:
    """All loops of k3, with no length bound."""
    return get_loops(k3_graph, max_length_m=float("inf"))


@pytest.fixture
def k4_graph() -> nx.Graph:
    """k4 with unit lengths."""
    return _graph_with_attrs(nx.complete_graph(4))


@pytest.fixture
def k4_loops(k4_graph) -> LoopData:
    """All loops of k4, with no length bound."""
    return get_loops(k4_graph, max_length_m=float("inf"))
