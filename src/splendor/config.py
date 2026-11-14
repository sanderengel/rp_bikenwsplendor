# Modeling parameters

LOOP_NUMNODE_BOUND = 30 # Maximum number of nodes in a loop
LOOP_LENGTH_BOUND = 40000 # Maximum total length of a loop in meters

CORRIDOR_RADIUS = 100  # Radius for link corridors in meters
ROAD_STRIP_RADIUS = 10 # Radius for link road strips in meters

EPSG = 25832

SEED = 70766

STUDY_AREA = "pilot"

EXCLUDED_TYPES = (13,) # 13 is UNCLASSIFIED (see src/splendor/landuse/types.py)

DIVERSITY_MEASURES = (
    # See src/splendor/diversity.py for supported functions
    "shannon",
    # "inverse_simpson", # Ranks almost identically to Shannon (see notebooks/exp/link_components.ipynb)
    "richness"
) 
RICHNESS_MIN_SHARE = .1

NORMALIZATION_METHODS = ("bounds", "percentile", "rank") # See src/splendor/composite/normalize.py
NORMALIZATION_PERCENTILE = 99

POI_BOUNDS = (0, 3) # Theoretical range of POI categories
GEOMETRIC_FLOOR = 0.25 # Components are lifted to floor + (1 - floor) * x in the geometric mean, so a 0 doesn't zero the score

AGGREGATION_METHODS = ("mean", "geometric") # See src/splendor/composite/aggregate.py
LAND_USE_WEIGHTS = (0.25, 0.5, 0.75) # Weight of land use in the score, POI gets 1 - w
REFERENCE_WEIGHTS = (0.0, 1.0) # Single-component rankings (POI only, land use only), kept out of the rank summary

MAIN_SPECIFICATION = {
    "measure": "shannon",
    "normalization": "percentile",
    "aggregation": "mean",
    "weight": 0.75
}
