"""
Validate that the pilot loops census agrees with bikenwloops.
"""
import pickle
import numpy as np
from pathlib import Path
from splendor.io import load_network
from splendor.loops import LoopData, get_loop_edge_ids, build_graph
from splendor.config import LOOP_NUMNODE_BOUND, LOOP_LENGTH_BOUND
from splendor.attributes import loop_max, loop_poi_diversity

# Load own graph and pilot census data
NETWORK = Path("data/interim/pilot/network.gpkg")
OWN = Path(f"data/processed/pilot/loops_{LOOP_NUMNODE_BOUND}_{LOOP_LENGTH_BOUND}.npz")
nodes, edges = load_network(NETWORK)
G = build_graph(nodes, edges)
loops = LoopData.load(OWN)


# Load pilot census data from bikenwloops
REF = Path(
    "../refs/bikenwloops/data/processed/pilot",
    f"loopcensus_{LOOP_NUMNODE_BOUND}_maxlength{LOOP_LENGTH_BOUND}.pkl"
)
with open(REF, "rb") as f:
    bwl_loops = pickle.load(f)
    bwl_loop_lengths = pickle.load(f)
    bwl_loop_n_nodes = pickle.load(f)
    bwl_loop_max_slopes = pickle.load(f)
    bwl_G = pickle.load(f)
    bwl_loop_numnode_bound = pickle.load(f)
    bwl_nodes_id = pickle.load(f)
    bwl_nodes_coords = pickle.load(f)
    bwl_numloops = pickle.load(f)

# Total node-loop pairs
print(f"bikenwloops total node-loop entries: {bwl_numloops}, bikenwsplendor total node-loop entries: {loops.sizes.sum()}")
assert loops.sizes.sum() == bwl_numloops

# Loops per node
bwl_per_node = np.array([len(bwl_loops[nid]["loops"]) for nid in range(loops.n_network_nodes)])
np.testing.assert_array_equal(bwl_per_node, loops.N.sum(axis=0))
print("Loops per node: identical")


# Identical loops, compared as sets of edges
def key(edge_ids) -> tuple[int, ...]:
    return tuple(sorted(edge_ids))

bwl_keys = {}
for nid, d in bwl_loops.items():
    for i, c in enumerate(d["loops"]):
        # Avoid counting same loops for all its nodes
        if c[0] != nid:
            continue

        eids = get_loop_edge_ids(G, c)
        bwl_keys[key(eids)] = (d["lengths"][i], d["max_slopes"][i], d["poi_diversity"][i])

own_keys = [key(loops.edges[a:b]) for a, b in zip(loops.ptr[:-1], loops.ptr[1:])]

assert len(bwl_keys) == loops.n_loops
assert set(bwl_keys) == set(own_keys)
print(f"Distinct loops: {loops.n_loops} identical sets")


order = [bwl_keys[k] for k in own_keys]
bwl_len, bwl_slope100, bwl_poi = (np.array(x) for x in zip(*order))

np.testing.assert_allclose(loops.lengths_m, bwl_len, rtol=1e6) # bikenwloops is float32
