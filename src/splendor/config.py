# Modeling parameters

LOOP_NUMNODE_BOUND = 30 # Maximum number of nodes in a loop
LOOP_LENGTH_BOUND = 40000 # Maximum total length of a loop in meters

CORRIDOR_RADIUS = 100  # Radius for link corridors in meters
ROAD_STRIP_RADIUS = 10 # Radius for link road strips in meters

EPSG = 25832

SEED = 70766

STUDY_AREA = "pilot"

EXCLUDED_TYPES = (13,) # 13 is UNCLASSIFIED (see src/splendor/landuse.py)

DIVERSITY_MEASURES = ("shannon", "inverse_simpson", "richness") # See src/splendor/score.py for supported functions

RICHNESS_MIN_SHARE = .1

NORMALIZATION_METHODS = ("bounds", "percentile", "rank") # See src/splendor/composite/normalize.py
NORMALIZATION_PERCENTILE = 99