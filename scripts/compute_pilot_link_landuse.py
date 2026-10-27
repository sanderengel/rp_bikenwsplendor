from splendor.io import load_network, load_raster
from splendor.landuse import N_TYPES, get_link_corridors, get_corridor_shares
from splendor.paths import network_path, basemap05_main_types_path, link_landuse_path
from splendor.config import CORRIDOR_RADIUS

STUDY_AREA = "pilot"
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
raster, transform, crs = load_raster(basemap05_main_types_path(STUDY_AREA))
assert crs == edges.crs, f"CRS mismatch: {crs} vs {edges.crs}"

corridors = get_link_corridors(edges, CORRIDOR_RADIUS)
corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
corridor_shares.insert(0, "edge_id", edges["edge_id"].to_numpy())

corridor_shares.to_csv(
    link_landuse_path(STUDY_AREA, CORRIDOR_RADIUS),
    index=False
)
