"""
Load the pilot network, build the graph and save loop data.
"""
import time
from splendor.io import load_network
from splendor.paths import network_path, loops_path
from splendor.loops import build_graph, get_loops
from splendor.config import LOOP_NUMNODE_BOUND, LOOP_LENGTH_BOUND

STUDY_AREA = "pilot"
SRC = network_path(STUDY_AREA)
DST = loops_path(STUDY_AREA, LOOP_NUMNODE_BOUND, LOOP_LENGTH_BOUND)

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
