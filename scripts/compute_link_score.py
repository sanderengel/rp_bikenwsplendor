"""Score every link under every combination of the grid, and summarize how sensitive its rank is to the choices."""
import pandas as pd

from splendor.io import load_network
from splendor.cli import get_study_area
from splendor.paths import (
    network_path,
    link_diversity_path,
    link_score_grid_path,
    link_rank_summary_path,
    link_rank_correlations_path,
    link_splendor_path
)
from splendor.config import (
    CORRIDOR_RADIUS,
    ROAD_STRIP_RADIUS,
    EXCLUDED_TYPES,
    DIVERSITY_MEASURES,
    RICHNESS_MIN_SHARE,
    NORMALIZATION_METHODS,
    NORMALIZATION_PERCENTILE,
    POI_BOUNDS,
    GEOMETRIC_FLOOR,
    AGGREGATION_METHODS,
    LAND_USE_WEIGHTS,
    REFERENCE_WEIGHTS,
    MAIN_SPECIFICATION
)
from splendor.landuse.types import N_TYPES
from splendor.diversity import get_bounds
from splendor.composite.sensitivity import get_score_grid, summarize_ranks, get_rank_correlations

STUDY_AREA = get_study_area()

# Components are land use diversity per measure plus POI diversity from the network
_, edges = load_network(network_path(STUDY_AREA))
diversity = pd.read_csv(link_diversity_path(STUDY_AREA, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS))

components = diversity.merge(
    edges[["edge_id", "poi_diversity"]], on="edge_id", how="inner", validate="one_to_one"
)
assert len(components) == len(edges) == len(diversity), "edge ids of network and diversity file don't match"

# Each component's theoretical bounds
n_types = N_TYPES - len(EXCLUDED_TYPES) # Types the measures see
bounds = {f"diversity_{m}": get_bounds(m, n_types, RICHNESS_MIN_SHARE) for m in DIVERSITY_MEASURES}
bounds["poi_diversity"] = POI_BOUNDS

# Every combination including the single-component reference weights
grid = get_score_grid(
    components,
    DIVERSITY_MEASURES,
    NORMALIZATION_METHODS,
    AGGREGATION_METHODS,
    land_use_weights=(*LAND_USE_WEIGHTS, *REFERENCE_WEIGHTS),
    bounds=bounds,
    p=NORMALIZATION_PERCENTILE,
    geometric_floor=GEOMETRIC_FLOOR,
)

# Rank summary over the combined scores only, but correlations over all combinations
summary = summarize_ranks(grid[grid["weight"].isin(LAND_USE_WEIGHTS)])
correlations = get_rank_correlations(grid)

# Filter to main specification to get final splendor scores
splendor = grid.copy()
for column, choice in MAIN_SPECIFICATION.items():
    splendor = splendor[splendor[column] == choice]
splendor = splendor[["edge_id", "score", "rank"]]
assert splendor["edge_id"].is_unique
assert set(splendor["edge_id"]) == set(edges["edge_id"]), "main specification must match exactly one combination in the grid"

path_args = (STUDY_AREA, CORRIDOR_RADIUS, ROAD_STRIP_RADIUS)
grid.to_csv(link_score_grid_path(*path_args), index=False)
summary.to_csv(link_rank_summary_path(*path_args), index=False)
correlations.to_csv(link_rank_correlations_path(*path_args))
splendor.to_csv(link_splendor_path(*path_args), index=False)
