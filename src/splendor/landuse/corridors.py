import numpy as np
import geopandas as gpd
from shapely import LineString, MultiPolygon, Polygon
from shapely.ops import substring, unary_union

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


def get_link_strips(edges: gpd.GeoDataFrame, r: float) -> gpd.GeoSeries:
    """Return a strip of all points within distance r of each link. Same index and CRS as the edges."""
    return edges.geometry.buffer(r)
