import numpy as np
import networkx as nx

from splendor.loops import LoopData, get_loops


def test_k3_graph(k3_loops):
    """The complete graph with 3 nodes should return 1 loop."""
    assert k3_loops.n_loops == 1

    # Unit edge lengths, loops should have length 3
    assert k3_loops.lengths_m[0] == 3


def test_k4_graph(k4_loops):
    """The complete graph with 4 nodes should return 7 loops (4 triangles + 3 squares)."""
    assert k4_loops.n_loops == 7

    # Each of the 4 nodes lies on 6 loops (3 triangles, 3 squares)
    np.testing.assert_array_equal(k4_loops.N.sum(axis=0), [6] * 4)

    # Each of the 6 edges lies on 4 loops (2 triangles, 2 squares)
    np.testing.assert_array_equal(k4_loops.M.sum(axis=0), [4] * 6)

    # Unit edge lengths so triangles have length 3, squares length 4
    np.testing.assert_array_equal(np.sort(k4_loops.lengths_m), [3, 3, 3, 3, 4, 4, 4])


def test_small_max_length(k4_graph):
    """A small max length should remove long loops."""
    loops = get_loops(k4_graph, max_length_m=3)

    assert loops.n_loops == 4
    np.testing.assert_array_equal(loops.lengths_m, [3] * 4)
    np.testing.assert_array_equal(loops.sizes, [3] * 4) # All 3-node loops


def test_save_load_roundtrip(k4_loops, tmp_path):
    """Loop data that is loaded should equal the original."""
    path = tmp_path / "loops.npz"
    k4_loops.save(path)
    loops_loaded = LoopData.load(path)

    for attr in ["nodes", "edges", "lengths_m", "ptr"]:
        np.testing.assert_array_equal(getattr(loops_loaded, attr), getattr(k4_loops, attr))
    
    assert loops_loaded.n_network_nodes == k4_loops.n_network_nodes
    assert loops_loaded.n_network_edges == k4_loops.n_network_edges
