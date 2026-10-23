import numpy as np
import geopandas as gpd
import networkx as nx
from collections.abc import Iterator
from scipy.sparse import csr_array
from pathlib import Path
from typing import Self

from splendor.config import LOOP_NUMNODE_BOUND, LOOP_LENGTH_BOUND


class LoopData:
    """Container class to store loop data for a set of loops"""
    def __init__(
        self,
        nodes: list,
        edges: list,
        lengths_m: list,
        ptr: list,
        n_network_nodes: int,
        n_network_edges: int,
    ) -> None:
        self.nodes = np.array(nodes, dtype=np.int32)
        self.edges = np.array(edges, dtype=np.int32)
        self.lengths_m = np.array(lengths_m, dtype=np.float64)
        self.ptr = np.array(ptr, dtype=np.int64)

        self.n_loops = len(ptr) - 1
        self.n_network_nodes = n_network_nodes
        self.n_network_edges = n_network_edges

        # Number of nodes for loop i is ptr[i+1] - ptr[i]
        self.sizes = np.diff(self.ptr)

        # Check IDs are 0...n-1
        assert self.nodes.max() < n_network_nodes or self.nodes.size == 0
        assert self.edges.max() < n_network_edges or self.edges.size == 0

        # Loop x edge incidence matrix
        self.M = csr_array((
            np.ones(len(self.edges), dtype=np.int8),
            self.edges, 
            self.ptr
        ), shape=(self.n_loops, n_network_edges))

        # Loop x node incidence matrix
        self.N = csr_array((
            np.ones(len(self.nodes), dtype=np.int8),
            self.nodes, 
            self.ptr
        ), shape=(self.n_loops, n_network_nodes))

    def save(self, path: str | Path) -> None:
        """Save loop data."""
        np.savez_compressed(
            path,
            nodes=self.nodes,
            edges=self.edges,
            lengths_m=self.lengths_m,
            ptr=self.ptr,
            n_network_nodes=self.n_network_nodes, # Stored as 0-d array
            n_network_edges=self.n_network_edges, # Stored as 0-d array
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load loop data saved with LoopData.save."""
        with np.load(path) as d:
            return cls(
                nodes=d["nodes"],
                edges=d["edges"],
                lengths_m=d["lengths_m"],
                ptr=d["ptr"],
                n_network_nodes=int(d["n_network_nodes"]), # Convert back from 0-d array
                n_network_edges=int(d["n_network_edges"]), # Convert back from 0-d array
            )


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


def get_loops(
    G: nx.Graph,
    max_nodes: int = LOOP_NUMNODE_BOUND,
    max_length_m: float = LOOP_LENGTH_BOUND
) -> LoopData:
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
        nodes=loop_nodes,
        edges=loop_edges,
        lengths_m=loop_lengths_m,
        ptr=loop_ptr,
        n_network_nodes=G.number_of_nodes(), # Take from G to include nodes not in any loops
        n_network_edges=G.number_of_edges()  # Take from G to include edges not in any loops
    )
