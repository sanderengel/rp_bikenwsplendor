import pytest
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from splendor.landuse import get_link_corridors, get_corridor_shares, N_TYPES


def test_corridors_keep_index(make_edges):
    """One corridor polygon per edge as a GeoSeries with the same index as the edges."""
    edges = make_edges(
        [(100, 800), (200, 800)], # 100m
        [(500, 200), (500, 800)]  # 600m
    ).set_index(pd.Index([10, 20])) # Custom index to check that it carries over

    corridors = get_link_corridors(edges, r=20)

    assert len(corridors) == len(edges)
    assert corridors.index.equals(edges.index)
    assert (corridors.geom_type == "Polygon").all()


def test_corridor_area(make_edges):
    """A straight line with flat caps has area of exactly 2rL."""
    L = 200
    x, y = 0, 0
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)

    assert round(corridors.area.iloc[0]) == 2 * r * L


def test_corridor_flat_caps(make_edges):
    """For a horizontal line, the corridor's x-range equals the line's x-range."""
    L = 200
    x, y = 0, 0
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)

    e_minx, _, e_maxx, _ = edges.geometry.bounds.iloc[0]
    c_minx, _, c_maxx, _ = corridors.bounds.iloc[0]

    assert e_minx == c_minx
    assert e_maxx == c_maxx


def test_shares_full_forest(half_raster, make_edges):
    "A line fully in forest area (code 4) should have 'share_4' == 1.0."
    L = 200
    x, y = 250, 100
    r = 20
    edges = make_edges([(x, y), (x, y + L)]) # Vertical
    corridors = get_link_corridors(edges, r=r)

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
    L = 200
    x, y = 500, 100
    r = 20
    edges = make_edges([(x, y), (x, y + L)]) # Vertical
    corridors = get_link_corridors(edges, r=r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row = corridor_shares.iloc[0]

    assert row["share_4"] == 0.5
    assert row["share_2"] == 0.5
    assert (row.drop(["n_pixels", "share_2", "share_4"]) == 0).all()


def test_shares_pixel_count(half_raster, make_edges):
    """A corridor on a straight line should have exactly 2rL / (xsize * ysize) pixels."""
    L = 200
    x, y = 250, 100
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)
    
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
    corridors = get_link_corridors(edges, r=r)
    
    raster, transform = random_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)
    row_sums = corridor_shares.drop(columns="n_pixels").sum(axis=1)

    np.testing.assert_allclose(row_sums, 1.0)


def test_shares_exclude_no_data(random_raster, make_edges):
    """Corridor excludes no-data pixels (type 0) in shares and n_pixels."""
    L = 200
    x, y = 100, 100
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)

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
    L = 200
    x, y = 250, 100
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)

    assert corridor_shares.columns.to_list() == ["n_pixels"] + [f"share_{k}" for k in range(1, N_TYPES + 1)]


def test_shares_no_data_error(nodata_raster, make_edges):
    """A corridor with no valid pixels should raise an AssertionError."""
    L = 200
    x, y = 250, 100
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r=r)

    raster, transform = nodata_raster
    with pytest.raises(AssertionError):
        get_corridor_shares(corridors, raster, transform, N_TYPES)
