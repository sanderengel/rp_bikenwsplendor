import numpy as np
import geopandas as gpd
import networkx as nx
from collections.abc import Iterator
from scipy.sparse import csr_array


class LoopData:
    """Container class to store loop data for a set of loops"""
    def __init__(
        self,
        loop_nodes: list,
        loop_edges: list,
        loop_lengths_m: list,
        loop_ptr: list,
        n_nodes: int,
        n_edges: int,
    ) -> None:
        # Check IDs are 0...n-1
        assert all(nid < n_nodes for nid in loop_nodes)
        assert all(eid < n_edges for eid in loop_edges)

        self.loop_nodes = np.array(loop_nodes, dtype=np.int32)
        self.loop_edges = np.array(loop_edges, dtype=np.int32)
        self.loop_lengths_m = np.array(loop_lengths_m, dtype=np.float64)
        self.loop_ptr = np.array(loop_ptr, dtype=np.int64)

        self.n_loops = len(loop_ptr) - 1
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        # Loop x edge incidence matrix
        self.M = csr_array((
            np.ones(len(self.loop_edges), dtype=np.int8),
            self.loop_edges, 
            self.loop_ptr
        ), shape=(self.n_loops, n_edges))

        # Loop x node incidence matrix
        self.N = csr_array((
            np.ones(len(self.loop_nodes), dtype=np.int8),
            self.loop_nodes, 
            self.loop_ptr
        ), shape=(self.n_loops, n_nodes))


def build_graph(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> nx.Graph:
    """Build undirected networkX graph with edge_id and length_m on each edge."""
    G = nx.Graph()
    G.add_nodes_from(nodes.node_id)
    G.add_edges_from(
        (u, v, {"edge_id": eid, "length_m": length})
        for u, v, eid, length in zip(edges.u, edges.v, edges.edge_id, edges.length_m)
    )
    return G


def _get_cycle_node_pairs(c: list) -> Iterator[tuple[int, int]]:
    n = len(c)
    for i in range(n):
        yield c[i], c[(i + 1) % n]


def _get_loop_edge_ids(G: nx.Graph, c: list) -> list[int]:
    return [G[u][v]["edge_id"] for u, v in _get_cycle_node_pairs(c)]


def _get_loop_length(G: nx.Graph, c: list) -> float:
    return sum(G[u][v]["length_m"] for u, v in _get_cycle_node_pairs(c))


def get_loops(G: nx.Graph, max_nodes: int, max_length_m: float) -> LoopData:
    """Get LoopData object from networkX graph."""
    # Store nodes and edges in concatenated arrays
    loop_nodes = []
    loop_edges = []

    loop_ptr = [0]      # Loop i has positions loop_ptr[i]:loop_ptr[i+1] in above lists
    loop_lengths_m = [] # One length value per loop

    for c in nx.simple_cycles(G, length_bound=max_nodes):
        loop_length_m = _get_loop_length(G, c)
        if loop_length_m > max_length_m:
            continue

        loop_nodes.extend(c)
        loop_edges.extend(_get_loop_edge_ids(G, c))
        loop_lengths_m.append(loop_length_m)
        loop_ptr.append(loop_ptr[-1] + len(c))

    # Return as loop data object
    return LoopData(
        loop_nodes=loop_nodes,
        loop_edges=loop_edges,
        loop_lengths_m=loop_lengths_m,
        loop_ptr=loop_ptr,
        n_nodes=G.number_of_nodes(), # Take from G to include nodes not in any loops
        n_edges=G.number_of_edges()  # Take from G to include edges not in any loops
    )
