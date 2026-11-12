import math
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Sequence
from affine import Affine
from rasterio.features import geometry_mask
from rasterio import windows
from shapely import LineString, MultiPolygon, Polygon, Geometry
from shapely.ops import substring, unary_union


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

CUT_EPS = 1.0        # Meters, cut disc slightly larger than r such that no sliver of the round cap survives
MIN_PART_AREA = 10.0 # Square meters, leftover fragments smaller than this are dropped

SHARE_PREFIX = "share_"


def _outward_direction(line: LineString, at_start: bool, s: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the link's end point and the unit vector pointing outward from it,
    measured over the last s meters of the link.
    """
    L = line.length
    end = np.array(line.coords[0] if at_start else line.coords[-1])
    inner = np.array(line.interpolate(s if at_start else L - s).coords[0])

    direction = end - inner
    norm = np.linalg.norm(direction)
    assert norm > 0, "zero-length direction"

    outward = direction / norm
    return end, outward


def _beyond_square(point: np.ndarray, outward: np.ndarray, size: float) -> Polygon:
    """
    Return the square on the far side of the line through the point, 
    perpendicular to the outward direction.
    """
    across = np.array([-outward[1], outward[0]]) # Perpendicular to outward, points along the line cutting though the point
    # Draw and return the square
    return Polygon([
        point + size * across,                      # From the point, size meters along the cutting line
        point + size * across + 2 * size * outward, # From there, 2*size meters outward
        point - size * across + 2 * size * outward, # Across to the other side, still 2*size outward
        point - size * across,                      # Back onto the cutting line, size meters to the other side of the point
    ])


def get_link_corridor(line: LineString, r: float) -> Polygon:
    """Corridor of one link: all points within r of it, minus what lies beyond its end nodes."""
    L = line.length
    s = min(r, L / 2)
    corridor = line.buffer(r) # All points within r distance from the line

    # At both ends, remove the part of the corridor beyond the end node: everything within r of the
    # link's last s metres that is on the far side of the cutting line
    for at_start in (True, False):
        endpoint, outward = _outward_direction(line, at_start, s)
        end_piece = substring(line, 0, s) if at_start else substring(line, L - s, L)
        beyond_square = _beyond_square(endpoint, outward, 2 * (r + s + CUT_EPS))
        cut = end_piece.buffer(r + CUT_EPS).intersection(beyond_square)
        corridor = corridor.difference(cut)

    # Drop numerical slivers
    if isinstance(corridor, MultiPolygon):
        corridor = unary_union([part for part in corridor.geoms if part.area >= MIN_PART_AREA])
    assert isinstance(corridor, Polygon), "corridor split into several parts"

    return corridor


def get_link_corridors(edges: gpd.GeoDataFrame, r: float) -> gpd.GeoSeries:
    """
    Return one corridor polygon per edge (link): all points within distance r of the link,
    i.e., the area swept by a disk of radius r along it. Ends cut perpendicular to the link's
    direction near each node.
    """
    return gpd.GeoSeries(
        [get_link_corridor(line, r) for line in edges.geometry],
        index=edges.index, crs=edges.crs
    )


def get_link_strips(edges: gpd.GeoDataFrame, r: float) -> gpd.GeoSeries:
    """Return a strip of all points within distance r of each link. Same index and CRS as the edges."""
    return edges.geometry.buffer(r)


def get_bounds_window(
        bounds: tuple[float, float, float, float], 
        transform: Affine, 
        raster_shape: tuple[int, int]
    ) -> windows.Window:
    """
    Return the pixel window covering the bounds, snapped outward to whole pixels so that 
    every pixel touched by the bounds is included.
    """
    bounds_window = windows.from_bounds(*bounds, transform=transform)
    r_start = math.floor(bounds_window.row_off)
    c_start = math.floor(bounds_window.col_off)
    r_stop = math.ceil(bounds_window.row_off + bounds_window.height)
    c_stop = math.ceil(bounds_window.col_off + bounds_window.width)

    assert_error_msg = "geom extends beyond the raster"
    assert r_start >= 0 and c_start >= 0, assert_error_msg
    assert r_stop <= raster_shape[0] and c_stop <= raster_shape[1], assert_error_msg

    window = windows.Window.from_slices((r_start, r_stop), (c_start, c_stop))
    return window


def get_pixel_values(geom: Geometry, raster: np.ndarray, transform: Affine) -> np.ndarray:
    """Return the raster values of all pixels whose center lies inside the geom."""
    window = get_bounds_window(geom.bounds, transform, raster.shape)
    window_values = raster[window.toslices()]

    # Geotransform of the window (same pixel size with origin shifted to the window's corner)
    window_transform = windows.transform(window, transform)

    mask = geometry_mask(
        [geom], # List of geometries, here is a single geom
        out_shape=window_values.shape,
        transform=window_transform,
        invert=True # True for pixels whose center is in the geom
    )

    return window_values[mask]


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
