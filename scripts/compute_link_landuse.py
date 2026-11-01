from splendor.io import load_network, load_raster
from splendor.cli import get_study_area
from splendor.landuse import N_TYPES, ROAD, get_link_corridors, get_link_strips, get_corridor_shares
from splendor.paths import network_path, basemap05_main_types_path, link_landuse_path
from splendor.config import CORRIDOR_RADIUS, ROAD_STRIP_RADIUS

STUDY_AREA = get_study_area()
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
raster, transform, crs = load_raster(basemap05_main_types_path(STUDY_AREA))
assert crs == edges.crs, f"CRS mismatch: {crs} vs {edges.crs}"

corridors = get_link_corridors(edges, CORRIDOR_RADIUS)
road_strips = get_link_strips(edges, ROAD_STRIP_RADIUS)
corridor_shares = get_corridor_shares(
    corridors, raster, transform, N_TYPES,
    strips=road_strips, strip_excluded_type=ROAD[0] # Ignore the link's own road
)
corridor_shares.insert(0, "edge_id", edges["edge_id"].to_numpy())

corridor_shares.to_csv(
    link_landuse_path(STUDY_AREA, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS),
    index=False
)
