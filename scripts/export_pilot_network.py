"""
Export the bikenwloops pilot network to a GeoPackage
"""
import lzma
import pickle
from pathlib import Path
from splendor.io import export_network, load_network
from splendor.config import BIKENWLOOPS_DIR

SRC = BIKENWLOOPS_DIR / "data/processed/pilot/network_preprocessed.xz"
DST = Path("data/interim/pilot/network.gpkg")

with lzma.open(SRC, "rb") as f:
    G = pickle.load(f)

DST.parent.mkdir(parents=True, exist_ok=True)
export_network(G, DST)
edges, nodes = load_network(DST)

# Checks
assert edges.edge_id.is_unique and nodes.node_id.is_unique
assert edges[["u", "v"]].isin(nodes.node_id.tolist()).all().all()
print(f"{len(nodes)} nodes, {len(edges)} edges, {edges.length_m.sum() / 1000:.0f} km")
print(edges.geom_type.value_counts())
