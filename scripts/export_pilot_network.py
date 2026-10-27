"""
Export the bikenwloops pilot network to a GeoPackage
"""
import lzma
import pickle
from splendor.io import export_network, load_network
from splendor.paths import bikenwloops_network_path, network_path

STUDY_AREA = "pilot"
SRC = bikenwloops_network_path(STUDY_AREA)
DST = network_path(STUDY_AREA)


with lzma.open(SRC, "rb") as f:
    G = pickle.load(f)

export_network(G, DST)
edges, nodes = load_network(DST)


# Checks
assert edges.edge_id.is_unique and nodes.node_id.is_unique
assert edges[["u", "v"]].isin(nodes.node_id.tolist()).all().all()
print(f"{len(nodes)} nodes, {len(edges)} edges, {edges.length_m.sum() / 1000:.0f} km")
print(edges.geom_type.value_counts())
