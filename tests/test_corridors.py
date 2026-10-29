import pytest
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from shapely import Point, Polygon, LineString, box

from splendor.landuse import get_link_corridor, get_link_corridors, get_corridor_shares, N_TYPES


def test_corridors_keep_metadata(make_edges):
    """One corridor polygon per edge as a GeoSeries with the same index and CRS as the edges."""
    r = 20
    edges = make_edges(
        [(100, 800), (200, 800)], # 100m
        [(500, 200), (500, 800)]  # 600m
    ).set_index(pd.Index([10, 20])) # Custom index to check that it carries over

    corridors = get_link_corridors(edges, r)

    assert len(corridors) == len(edges)
    assert corridors.index.equals(edges.index)
    assert corridors.crs == edges.crs


def test_corridor_area(make_edges):
    """A straight line with flat caps has area of exactly 2rL."""
    x, y = 0, 0
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)

    assert round(corridors.area.iloc[0]) == 2 * r * L


def test_corridor_flat_caps(make_edges):
    """For a horizontal line, the corridor's x-range equals the line's x-range."""
    x, y = 0, 0
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)

    e_minx, _, e_maxx, _ = edges.geometry.bounds.iloc[0]
    c_minx, _, c_maxx, _ = corridors.bounds.iloc[0]

    assert e_minx == c_minx
    assert e_maxx == c_maxx


def test_corridors_straight_line_is_rectangle():
    """A straight line's corridor is exactly the 2rL rectangle around it."""
    x, y = 0, 0
    L = 200
    r = 20
    line = LineString([(x, y), (x + L, y)])
    corridor = get_link_corridor(line, r)
    rectangle = box(x, y - r, x + L, y + r)

    assert corridor.symmetric_difference(rectangle).area == pytest.approx(0, abs=1e-6)


def test_corridors_hooked_and_keeps_side():
    """
    A short hook at the end shouldn't tilt the cut. A point within r of the main segment and
    next to the road near the end should stay inside.
    """
    x, y = 0, 0
    xL, yL = 300, 5
    r = 50
    line = LineString([(x, y), (x + xL, y), (x + xL, y + yL)]) # xL m east, then a yL m hook at 45 degrees
    corridor = get_link_corridor(line, r)

    assert corridor.contains(Point(290, 45))


def test_corridors_exclude_beyond_node():
    """Nothing past the end node, in the link's overall direction, is in the corridor."""
    x, y = 0, 0
    xL, yL = 300, 5
    r = 50
    line = LineString([(x, y), (x + xL, y), (x + xL, y + yL)]) # xL m east, then a yL m hook at 45 degrees
    corridor = get_link_corridor(line, r)

    assert not corridor.contains(Point(330, 0))


def test_corridors_u_shape_not_cut():
    """For a U-shaped link, cutting one end must not remove corridor alongside the other leg."""
    x, y = 0, 0
    xL, yL = 40, 100
    r = 20
    line = LineString([(x, y), (x, y + yL), (x + xL, y + yL), (x + xL, y)])
    corridor = get_link_corridor(line, r)

    assert isinstance(corridor, Polygon)
    assert corridor.contains(Point(55, 50)) # Beside the far leg, within r of it


def test_corridors_hooked_end_no_tails():
    """Nothing of the corridor lies beyond the end node, not even at the corners of one that's hooked."""
    x, y = 0, 0
    xL, yL = 300, 20
    r = 50
    line = LineString([(x, y), (x + xL, y), (x + xL, y + yL)]) # 20 m hook at 90 degrees
    corridor = get_link_corridor(line, r)

    # Outward direction over the last r meters: from r meters back along the line to the end
    end = np.array([x + xL, y + yL])
    inner = np.array([x + xL - (r - yL), y]) # On the horizontal segment, r - yL before the corner
    outward = (end - inner) / np.linalg.norm(end - inner) # The road's overall direction over its last r meters
    across = np.array([-outward[1], outward[0]])

    # Large region on the far of the cutting line through the end node
    size = 10 * r
    beyond_square = Polygon([
        end + size * across,                      # From the end, size meters along the cutting line
        end + size * across + 2 * size * outward, # From there, 2*size meters outward
        end - size * across + 2 * size * outward, # Across to the other side, still 2*size outward
        end - size * across,                      # Back onto the cutting line, size meters to the other side of the point
    ])

    assert corridor.intersection(beyond_square).area == pytest.approx(0, abs=1e-6)


def test_shares_full_forest(half_raster, make_edges):
    "A line fully in forest area (code 4) should have 'share_4' == 1.0."
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x, y + L)]) # Vertical
    corridors = get_link_corridors(edges, r)

    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row = corridor_shares.iloc[0]

    assert row["share_4"] == 1.0
    assert (row.drop(["n_pixels", "share_4"]) == 0).all()


def test_shares_on_boundary(half_raster, make_edges):
    """
    A line on the boundary of the half-and-half raster should have 
    'share_4' == 0.5 and 'share_2' == 0.5.
    """
    x, y = 500, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x, y + L)]) # Vertical
    corridors = get_link_corridors(edges, r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row = corridor_shares.iloc[0]

    assert row["share_4"] == 0.5
    assert row["share_2"] == 0.5
    assert (row.drop(["n_pixels", "share_2", "share_4"]) == 0).all()


def test_shares_pixel_count(half_raster, make_edges):
    """A corridor on a straight line should have exactly 2rL / (xsize * ysize) pixels."""
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    n_pixels = corridor_shares.at[0, "n_pixels"]

    xsize = transform.a  # Pixel width
    ysize = -transform.e # Pixel height (stored as negative, since y decreases downwards)

    assert n_pixels == 2 * r * L / (xsize * ysize)


def test_shares_sum(random_raster, make_edges):
    """Corridor shares sum to 1.0."""
    r = 20
    edges = make_edges(
        [(250, 100), (450, 100)], # Horizontal
        [(100, 300), (400, 700)], # Diagonal
        [(600, 200), (600, 800)]  # Vertical
    )
    corridors = get_link_corridors(edges, r)
    
    raster, transform = random_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row_sums = corridor_shares.drop(columns="n_pixels").sum(axis=1)

    np.testing.assert_allclose(row_sums, 1.0)


def test_shares_exclude_no_data(random_raster, make_edges):
    """Corridor excludes no-data pixels (type 0) in shares and n_pixels."""
    x, y = 100, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)

    # Random raster randomly includes no-data pixels
    raster, transform = random_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row = corridor_shares.iloc[0]

    pixel = transform.a
    assert all(v % pixel == 0 for v in (x, y, L, r)), "test assumes coordinates on pixel edges"

    minx, miny, maxx, maxy = corridors.iloc[0].bounds
    r_start, c_start = rowcol(transform, minx, maxy) # Corridor's top-left pixel (row, col)
    r_stop, c_stop = rowcol(transform, maxx, miny)   # One past its bottom-right pixel (exclusive stop)
    
    block = raster[r_start:r_stop, c_start:c_stop]
    assert (block == 0).any(), "test needs some no-data pixels in the corridor"

    valid = block[block != 0]

    expected_shares = np.bincount(valid, minlength=N_TYPES + 1)[1:] / valid.size

    assert row["n_pixels"] == valid.size
    np.testing.assert_allclose(row.drop("n_pixels").to_numpy(dtype=float), expected_shares)
    assert row.drop("n_pixels").sum() == pytest.approx(1)


def test_shares_keep_index(random_raster, make_edges):
    """The corridor shares DataFrame should have one row per corridor with the corridors' index."""
    edges = make_edges(
        [(100, 800), (200, 800)], # 100m
        [(500, 200), (500, 800)]  # 600m
    ).set_index(pd.Index([10, 20])) # Custom index to check that it carries over
    corridors = get_link_corridors(edges, r=20)

    raster, transform = random_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)

    assert corridor_shares.index.equals(corridors.index)
    # Rows stay matched to their corridors (the longer corridor has more pixels)
    assert corridor_shares.at[20, "n_pixels"] > corridor_shares.at[10, "n_pixels"]


def test_shares_columns(half_raster, make_edges):
    """
    The corridor shares DataFrame should contain columns 'n_pixels' followed by 'share_1' ... 'share_N',
    with all types present even when their share is 0.
    """
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)

    assert corridor_shares.columns.to_list() == ["n_pixels"] + [f"share_{k}" for k in range(1, N_TYPES + 1)]


def test_shares_no_data_error(nodata_raster, make_edges):
    """A corridor with no valid pixels should raise an AssertionError."""
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)

    raster, transform = nodata_raster
    with pytest.raises(AssertionError):
        get_corridor_shares(corridors, raster, transform, N_TYPES)
