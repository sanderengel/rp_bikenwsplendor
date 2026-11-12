import pytest
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from splendor.landuse.corridors import get_link_strips, get_link_corridors
from splendor.landuse.shares import get_share_column_name, get_corridor_shares, get_shares_array
from splendor.landuse.types import N_TYPES, ROAD, UNCLASSIFIED


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
    share4_colname = get_share_column_name(4)

    assert row[share4_colname] == 1.0
    assert (get_shares_array(row.drop([share4_colname])) == 0).all()


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
    share4_colname = get_share_column_name(4)
    share2_colname = get_share_column_name(2)

    assert row[share4_colname] == 0.5
    assert row[share2_colname] == 0.5
    assert (get_shares_array(row.drop([share2_colname, share4_colname])) == 0).all()


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
    row_sums = get_shares_array(corridor_shares).sum(axis=1)

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
    corridor_shares_arr = get_shares_array(row)
    np.testing.assert_allclose(corridor_shares_arr, expected_shares)
    assert corridor_shares_arr.sum() == pytest.approx(1)


def test_shares_keep_index(random_raster, make_edges):
    """The corridor shares DataFrame should have one row per corridor with the corridors' index."""
    r = 20
    edges = make_edges(
        [(100, 800), (200, 800)], # 100m
        [(500, 200), (500, 800)]  # 600m
    ).set_index(pd.Index([10, 20])) # Custom index to check that it carries over
    corridors = get_link_corridors(edges, r)

    raster, transform = random_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)

    assert corridor_shares.index.equals(corridors.index)
    # Rows stay matched to their corridors (the longer corridor has more pixels)
    assert corridor_shares.at[20, "n_pixels"] > corridor_shares.at[10, "n_pixels"]


def test_shares_columns(half_raster, make_edges):
    """
    The corridor shares DataFrame should contain columns 'n_pixels' and 'n_excluded', 
    followed by 'share_1' ... 'share_N', with all types present even when their share is 0.
    """
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)]) # Horizontal
    corridors = get_link_corridors(edges, r)
    
    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES)

    # Testing the schema itself, so we don't use get_shares_array or get_share_column_name
    share_column_names = [f"share_{k}" for k in range(1, N_TYPES + 1)]
    assert corridor_shares.columns.to_list() == ["n_pixels", "n_excluded"] + share_column_names


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


def test_shares_ignore_road_in_strip(road_raster, make_edges):
    """A link on the road ignores the road is all forest."""
    x, y = 500, 100
    L = 600
    r = 50
    r_s = 15
    edges = make_edges([(x, y), (x, y + L)]) # Vertical on the road column's left edge
    corridors = get_link_corridors(edges, r)
    strips = get_link_strips(edges, r_s)

    raster, transform = road_raster
    corridor_shares = get_corridor_shares(
        corridors, raster, transform, N_TYPES,
        strips=strips, strip_excluded_type=ROAD[0] 
    )
    row = corridor_shares.iloc[0]

    assert row[get_share_column_name(4)] == 1.0
    assert row[get_share_column_name(ROAD[0])] == 0


def test_shares_keep_road_outside_strip(road_raster, make_edges):
    """A link 30 m from the road keeps it: the road is 1 of the corridor's 10 pixel columns."""
    x, y = 470, 100
    L = 600
    r = 50
    r_s = 15
    edges = make_edges([(x, y), (x, y + L)]) # Vertical 30 m left of the road column
    corridors = get_link_corridors(edges, r)
    strips = get_link_strips(edges, r_s)

    raster, transform = road_raster
    corridor_shares = get_corridor_shares(
        corridors, raster, transform, N_TYPES,
        strips=strips, strip_excluded_type=ROAD[0] 
    )
    row = corridor_shares.iloc[0]

    assert row[get_share_column_name(4)] == pytest.approx(0.9)
    assert row[get_share_column_name(ROAD[0])] == pytest.approx(0.1)


def test_shares_strip_and_rest_cover_corridor(random_raster, make_edges):
    """
    With strips but an excluded type that doesn't occur in the raster, nothing is dropped,
    so the result equals the one without strips, i.e., strip + rest pixels are exactly the corridor's.
    """
    r = 37
    r_s = 12
    edges = make_edges(
        [(123.4, 217.9), (561.3, 688.2)], # Diagonal, off the pixel grid
        [(250, 100), (450, 100)]          # Horizontal
    )
    corridors = get_link_corridors(edges, r)
    strips = get_link_strips(edges, r_s)

    raster, transform = random_raster
    absent_type = N_TYPES + 1 # Random raster only holds values between 0 and N_TYPES
    assert not (raster == absent_type).any(), "test needs a type that isn't in the raster"

    without_strips = get_corridor_shares(corridors, raster, transform, N_TYPES)
    with_strips = get_corridor_shares(
        corridors, raster, transform, N_TYPES, 
        strips=strips, strip_excluded_type=absent_type
    )

    pd.testing.assert_frame_equal(with_strips, without_strips)


def test_shares_strips_without_excluded_type_error(half_raster, make_edges):
    """Strips without a type to exclude should raise a ValueError."""
    x, y = 250, 100
    L = 200
    r = 20
    r_s = 10
    edges = make_edges([(x, y), (x + L, y)])
    corridors = get_link_corridors(edges, r)
    strips = get_link_strips(edges, r_s)

    raster, transform = half_raster
    with pytest.raises(ValueError):
        get_corridor_shares(corridors, raster, transform, N_TYPES, strips=strips)


def test_shares_strips_index_mismatch_error(half_raster, make_edges):
    """Strips whose index doesn't match the corridors' should raise a ValueError."""
    x, y = 250, 100
    L = 200
    r = 20
    r_s = 10
    edges = make_edges([(x, y), (x + L, y)])
    corridors = get_link_corridors(edges, r)
    strips = get_link_strips(edges, r_s).set_axis([99]) # Different index

    raster, transform = half_raster
    with pytest.raises(ValueError):
        get_corridor_shares(
            corridors, raster, transform, N_TYPES, 
            strips=strips, strip_excluded_type=4
        )


def test_shares_exclude_type(half_raster, make_edges):
    """
    Excluding cropland (code 2) on the forest/cropland boundary leaves only forest,
    with cropland pixels counted in 'n_excluded'.
    """
    x, y = 500, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x, y + L)])
    corridors = get_link_corridors(edges, r)

    raster, transform = half_raster
    corridor_shares = get_corridor_shares(
        corridors, raster, transform, N_TYPES,
        excluded_types=(2,)    
    )
    row = corridor_shares.iloc[0]

    assert get_share_column_name(2) not in corridor_shares.columns
    assert row[get_share_column_name(4)] == 1.0
    assert row["n_pixels"] == row["n_excluded"] # Half forest half cropland


def test_shares_exclude_equals_rescaled(random_raster, make_edges):
    """
    Excluding a type equals dropping its share and rescaling the rest by 1 / (1 - its share),
    and n_pixels + n_excluded equals n_pixels without exclusion.
    """
    r = 37
    edges = make_edges(
        [(123.4, 217.9), (561.3, 688.2)], # Diagonal, off the pixel grid
        [(250, 100), (450, 100)]          # Horizontal
    )
    corridors = get_link_corridors(edges, r)
    excluded = UNCLASSIFIED[0]
    excluded_col = get_share_column_name(excluded)

    raster, transform = random_raster
    without = get_corridor_shares(corridors, raster, transform, N_TYPES)
    with_excl = get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=(excluded,))
    assert (without[excluded_col] > 0).all(), "test needs the excluded type in every corridor"

    expected = (
        get_shares_array(without.drop(columns=excluded_col))
        / (1 - without[excluded_col].to_numpy())[:, None]
    )
    np.testing.assert_allclose(get_shares_array(with_excl), expected)
    np.testing.assert_array_equal(with_excl["n_pixels"] + with_excl["n_excluded"], without["n_pixels"])
    assert (without["n_excluded"] == 0).all()


def test_shares_exclude_columns(half_raster, make_edges):
    """Share columns are left out for excluded types and the rest keep their order."""
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)])
    corridors = get_link_corridors(edges, r)

    raster, transform = half_raster
    corridor_shares = get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=(13,))

    # Testing the schema itself, so literal names
    share_column_names = [f"share_{k}" for k in range(1, N_TYPES + 1) if k != 13]
    assert corridor_shares.columns.to_list() == ["n_pixels", "n_excluded"] + share_column_names


def test_shares_exclude_duplicates(random_raster, make_edges):
    """Duplicate excluded types are counted once."""
    r = 37
    edges = make_edges([(250, 100), (450, 100)])
    corridors = get_link_corridors(edges, r)

    raster, transform = random_raster
    once = get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=(13,))
    twice = get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=(13, 13))

    pd.testing.assert_frame_equal(twice, once)


@pytest.mark.parametrize("excluded_types", [(0,), (N_TYPES + 1,), (-1,)])
def test_shares_exclude_out_of_range_error(half_raster, make_edges, excluded_types):
    """Excluded types outside 1..N_TYPES should raise a ValueError."""
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x + L, y)])
    corridors = get_link_corridors(edges, r)

    raster, transform = half_raster
    with pytest.raises(ValueError):
        get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=excluded_types)


def test_shares_all_excluded_error(half_raster, make_edges):
    """A corridor whose pixels are all of excluded types should raise an AssertionError."""
    x, y = 250, 100
    L = 200
    r = 20
    edges = make_edges([(x, y), (x, y + L)]) # Vertical, fully in forest
    corridors = get_link_corridors(edges, r)

    raster, transform = half_raster
    with pytest.raises(AssertionError):
        get_corridor_shares(corridors, raster, transform, N_TYPES, excluded_types=(4,))
