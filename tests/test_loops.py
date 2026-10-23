import numpy as np
import networkx as nx

from splendor.loops import LoopData, get_loops


def _graph_with_attrs(G: nx.Graph, length_m: float = 1.) -> nx.Graph:
    """Add get_loops' expected edge_id and length_m attributes."""
    for eid, (u, v) in enumerate(G.edges()):
        G[u][v]["edge_id"] = eid
        G[u][v]["length_m"] = length_m
    return G


def test_k3_graph():
    """The complete graph with 3 nodes should return 1 loop."""
    G = _graph_with_attrs(nx.complete_graph(3))
    loops = get_loops(G, max_length_m=float("inf"))

    assert loops.n_loops == 1

    # Unit edge lengths, loops should have length 3
    assert loops.lengths_m[0] == 3


def test_k4_graph():
    """The complete graph with 4 nodes should return 7 loops (4 triangles + 3 squares)."""
    G = _graph_with_attrs(nx.complete_graph(4))
    loops = get_loops(G, max_length_m=float("inf"))

    assert loops.n_loops == 7

    # Each of the 4 nodes lies on 6 loops (3 triangles, 3 squares)
    np.testing.assert_array_equal(loops.N.sum(axis=0), [6] * 4)

    # Each of the 6 edges lies on 4 loops (2 triangles, 2 squares)
    np.testing.assert_array_equal(loops.M.sum(axis=0), [4] * 6)

    # Unit edge lengths so triangles have length 3, squares length 4
    np.testing.assert_array_equal(np.sort(loops.lengths_m), [3, 3, 3, 3, 4, 4, 4])


def test_small_max_length():
    """A small max length should remove long loops."""
    G = _graph_with_attrs(nx.complete_graph(4))
    loops = get_loops(G, max_length_m=3)

    assert loops.n_loops == 4
    np.testing.assert_array_equal(loops.lengths_m, [3] * 4)
    np.testing.assert_array_equal(loops.sizes, [3] * 4) # All 3-node loops


def test_save_load_roundtrip(tmp_path):
    """Loop data that is loaded should equal the original."""
    G = _graph_with_attrs(nx.complete_graph(4))
    loops_orig = get_loops(G, max_length_m=3)

    path = tmp_path / "loops.npz"
    loops_orig.save(path)
    loops_loaded = LoopData.load(path)

    for attr in ["nodes", "edges", "lengths_m", "ptr"]:
        np.testing.assert_array_equal(getattr(loops_loaded, attr), getattr(loops_orig, attr))
    
    assert loops_loaded.n_network_nodes == loops_orig.n_network_nodes
    assert loops_loaded.n_network_edges == loops_orig.n_network_edges
