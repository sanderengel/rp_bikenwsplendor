# Main types are the first three digits of the C_26 code (id, name, short name, color)
MAIN_TYPES = {
    101: (1, "Settlement and infrastructure", "Settlement", (200, 60, 60)),
    102: (2, "Cropland", "Cropland", (240, 220, 120)),
    103: (3, "Grassland", "Grassland", (170, 220, 120)),
    104: (4, "Forest", "Forest", (40, 120, 50)),
    105: (5, "Heathland and shrub", "Heathland", (150, 90, 160)),
    106: (6, "Sparsely vegetated", "Sparse veg.", (200, 200, 190)),
    107: (7, "Mires, fens and bogs", "Mires", (120, 170, 160)),
    108: (8, "Rivers", "Rivers", (60, 120, 220)),
    109: (9, "Lakes", "Lakes", (60, 120, 220)),
    110: (10, "Lagoons and tidal flats", "Lagoons", (120, 190, 230)),
    111: (11, "Coast (dunes, beaches, saltmarsh)", "Coast", (240, 200, 160)),
    112: (12, "Sea", "Sea", (30, 70, 160))
}
UNCLASSIFIED = (13, "Terrestrial, not classified", "Unclassified", (130, 130, 130))

# Roads and rail are split out of settlement by their full C_26 code
ROAD_CODE = 101301
ROAD = (14, "Road and rail networks", "Road/rail", (70, 70, 70))

ALL_TYPES = [*MAIN_TYPES.values(), UNCLASSIFIED, ROAD]
N_TYPES = len(ALL_TYPES)
SHORT_NAMES = {t[0]: t[2] for t in ALL_TYPES}
