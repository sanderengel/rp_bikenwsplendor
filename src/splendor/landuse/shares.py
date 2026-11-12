import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Sequence
from affine import Affine
from splendor.landuse.raster import get_pixel_values

SHARE_PREFIX = "share_"


def get_share_column_name(k: int) -> str:
    return SHARE_PREFIX + str(k)


def get_corridor_shares(
    corridors: gpd.GeoSeries,
    raster: np.ndarray,
    transform: Affine, 
    n_types: int,
    strips: gpd.GeoSeries | None = None,
    strip_excluded_type: int | None = None,
    excluded_types: Sequence[int] = ()
) -> pd.DataFrame:
    """
    Return the pixel value shares for each corridor from the raster data.
    Treat types in 'excluded_types' as no data by leaving them out of the shares,
    though their pixels are counted in the column 'n_excluded'.
    """
    excluded_types = np.unique(np.asarray(excluded_types, dtype=int))
    if ((excluded_types < 1) | (excluded_types > n_types)).any():
        raise ValueError(f"excluded types must be in 1..{n_types}")
    if strips is not None:
        if strip_excluded_type is None:
            raise ValueError("strips given without a type to exclude")
        if not strips.index.equals(corridors.index):
            raise ValueError("strips must have the same index as the corridors")

    # Initialize counts, +1 type so column 0 represents no data
    counts = np.zeros((len(corridors), n_types + 1), dtype=np.int64)

    for i, corridor in enumerate(corridors):
        if strips is None:
            values = get_pixel_values(corridor, raster, transform)
        else:
            strip = strips.iloc[i]
            # Split the corridor into the part outside and inside the strip
            rest_values = get_pixel_values(corridor.difference(strip), raster, transform)
            strip_values = get_pixel_values(corridor.intersection(strip), raster, transform)
            strip_values = strip_values[strip_values != strip_excluded_type]
            values = np.concatenate([rest_values, strip_values])
        counts[i] = np.bincount(values, minlength=n_types + 1)

    kept = np.ones(n_types + 1, dtype=bool)
    kept[0] = False        # No data
    kept[excluded_types] = False # Excluded

    n_pixels = counts[:, kept].sum(axis=1)
    assert (n_pixels > 0).all(), "corridor with no land use pixels"
    shares = counts[:, kept] / n_pixels[:, None] # Normalize each row

    kept_types = np.flatnonzero(kept)
    corridor_shares = pd.DataFrame(
        shares,
        columns = [get_share_column_name(k) for k in kept_types],
        index = corridors.index
    )
    corridor_shares.insert(0, "n_pixels", n_pixels)
    corridor_shares.insert(1, "n_excluded", counts[:, excluded_types].sum(axis=1))
    return corridor_shares


def get_shares_array(shares: pd.DataFrame | pd.Series) -> np.ndarray:
    """Return array of only share columns, stripping identifiers and aggregate columns."""
    return shares.filter(regex=f"^{SHARE_PREFIX}").to_numpy()
