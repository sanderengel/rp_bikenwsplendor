import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from rasterio.features import geometry_mask
from rasterio import windows


# Main types are the first three digits of the C_26 code (id, name, color)
MAIN_TYPES = {
    101: (1, "Settlement and infrastructure", (200, 60, 60)),
    102: (2, "Cropland", (240, 220, 120)),
    103: (3, "Grassland", (170, 220, 120)),
    104: (4, "Forest", (40, 120, 50)),
    105: (5, "Heathland and shrub", (150, 90, 160)),
    106: (6, "Sparsely vegetated", (200, 200, 190)),
    107: (7, "Mires, fens and bogs", (120, 170, 160)),
    108: (8, "Rivers", (60, 120, 220)),
    109: (9, "Lakes", (60, 120, 220)),
    110: (10, "Lagoons and tidal flats", (120, 190, 230)),
    111: (11, "Coast (dunes, beaches, saltmarsh)", (240, 200, 160)),
    112: (12, "Sea", (30, 70, 160))
}
UNCLASSIFIED = (13, "Terrestrial, not classified", (130, 130, 130))
ALL_TYPES = [*MAIN_TYPES.values(), UNCLASSIFIED]
N_TYPES = len(ALL_TYPES)


def get_link_corridors(edges: gpd.GeoDataFrame, r: float) -> gpd.GeoSeries:
    """
    Return one corridor polygon per edge (link): all points within distance r of the link,
    i.e., the area swept by a disk of radius r along it. Flat end caps, so each corridor
    ends at the link's nodes. Same index as 'edges'.
    """
    return edges.geometry.buffer(r, cap_style="flat")


def get_corridor_shares(
    corridors: gpd.GeoSeries,
    raster: np.ndarray,
    transform: Affine, 
    n_types: int
) -> pd.DataFrame:
    """
    Return the pixel value shares for each corridor from the raster data.
    """
    # Initialize counts, +1 type so column 0 represents no data
    counts = np.zeros((len(corridors), n_types + 1), dtype=np.int64)

    for i, corridor in enumerate(corridors):
        # Build pixel window around the corridor
        window = (
            windows.from_bounds(*corridor.bounds, transform=transform)
            .round_offsets().round_lengths() # Snap to whole pixels
        )
        window_values = raster[window.toslices()]

        # Geotransform of the window (same pixel size with origin shifted to the window's corner)
        window_transform = windows.transform(window, transform)

        mask = geometry_mask(
            [corridor], # List of geometries, here is a single corridor
            out_shape=window_values.shape,
            transform=window_transform,
            invert=True # True for pixels whose centre is in the corridor
        )

        # Get pixel counts per type inside the corridor
        values = window_values[mask]
        counts[i] = np.bincount(values, minlength=n_types + 1)

    n_pixels = counts[:, 1:].sum(axis=1)       # Exclude no data (type 0)
    assert (n_pixels > 0).all(), "corridor with no land use pixels"
    shares = counts[:, 1:] / n_pixels[:, None] # Normalize each row

    corridor_shares = pd.DataFrame(
        shares,
        columns = [f"share_{k}" for k in range(1, n_types + 1)],
        index = corridors.index
    )
    corridor_shares.insert(0, "n_pixels", n_pixels)
    return corridor_shares
