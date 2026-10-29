import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from rasterio.features import geometry_mask
from rasterio import windows
from shapely import LineString, MultiPolygon, Polygon
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
ALL_TYPES = [*MAIN_TYPES.values(), UNCLASSIFIED]
N_TYPES = len(ALL_TYPES)
SHORT_NAMES = {t[0]: t[2] for t in ALL_TYPES}


CUT_EPS = 1.0        # Meters, cut disc slightly larger than r such that no sliver of the round cap survives
MIN_PART_AREA = 10.0 # Square meters, leftover fragments smaller than this are dropped


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
