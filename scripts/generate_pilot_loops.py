"""
Load the pilot network, build the graph and save loop data.
"""
import time
from pathlib import Path
from splendor.io import load_network
from splendor.loops import build_graph, get_loops
from splendor.config import LOOP_NUMNODE_BOUND, LOOP_LENGTH_BOUND

SRC = Path("data/interim/pilot/network.gpkg")
DST = Path(f"data/processed/pilot/loops_{LOOP_NUMNODE_BOUND}_{LOOP_LENGTH_BOUND}.npz")
DST.parent.mkdir(parents=True, exist_ok=True)

print(f"Generating pilot loops using network from '{SRC}'...")
t0 = time.perf_counter()

nodes, edges = load_network(SRC)
G = build_graph(nodes, edges)
loops = get_loops(G, max_nodes=LOOP_NUMNODE_BOUND, max_length_m=LOOP_LENGTH_BOUND)
loops.save(DST)

t1 = time.perf_counter()
print(f"Saved loop data to '{DST}'.")
print(f"Number of distinct loops: {loops.n_loops}")
print(f"Total number of loop nodes: {loops.sizes.sum()}")
print(f"Run time: {(t1 - t0):.2f}s")
