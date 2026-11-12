import pandas as pd

from splendor.io import load_network
from splendor.cli import get_study_area
from splendor.paths import network_path, link_landuse_path, link_diversity_path
from splendor.landuse import get_shares_array
from splendor.config import CORRIDOR_RADIUS, ROAD_STRIP_RADIUS, DIVERSITY_MEASURES
from splendor.diversity import get_diversity

STUDY_AREA = get_study_area()
NETWORK = network_path(STUDY_AREA)

_, edges = load_network(NETWORK)
corridor_shares = pd.read_csv(link_landuse_path(STUDY_AREA, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS))
assert edges["edge_id"].equals(corridor_shares["edge_id"])

shares_array = get_shares_array(corridor_shares)

link_diversity = pd.DataFrame({
    "edge_id": corridor_shares["edge_id"],
    **{f"diversity_{m}": get_diversity(shares_array, m) for m in DIVERSITY_MEASURES},
})

link_diversity.to_csv(
    link_diversity_path(STUDY_AREA, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS),
    index=False,
)
