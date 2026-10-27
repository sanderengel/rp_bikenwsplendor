import geopandas as gpd
import igraph as ig
from pathlib import Path
from shapely.geometry import Point

from splendor.paths import ensure_parent

CRS = 25832
DROP = {"geometry", "node_start", "node_end"} # stale ids from bikenwloops


def export_network(G: ig.Graph, path: str | Path) -> None:
    path = ensure_parent(path)

    # Nodes, ids equal to igraph vertex index (u/v before)
    nodes = gpd.GeoDataFrame(
        {
            "node_id": range(G.vcount()),
            **{a: G.vs[a] for a in G.vs.attributes() if a not in {"x", "y"}}
        },
        geometry=[Point(x, y) for x, y in zip(G.vs["x"], G.vs["y"])],
        crs=CRS
    )

    # Edges, these have own stable ids 0 to m-1, keep them for traceability
    e_attrs = {a: G.es[a] for a in G.es.attributes() if a not in DROP}
    edges = gpd.GeoDataFrame(
        {
            "edge_id": range(G.ecount()),
            "u": [e.source for e in G.es],
            "v": [e.target for e in G.es],
            **{("orig_edge_id" if a == "edge_id" else a): vals for a, vals in e_attrs.items()}
        },
        geometry=G.es["geometry"],
        crs=CRS,
    ).rename(columns={"weight": "length_m"})

    nodes.to_file(path, layer="nodes", driver="GPKG")
    edges.to_file(path, layer="edges", driver="GPKG")


def load_network(path: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """load the nodes and edges layers of a network GeoPackage."""
    nodes = gpd.read_file(path, layer="nodes")
    edges = gpd.read_file(path, layer="edges")
    return nodes, edges
