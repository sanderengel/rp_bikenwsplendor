import pytest
import numpy as np
import networkx as nx
import geopandas as gpd
import rasterio
from affine import Affine
from shapely.geometry import LineString
from typing import Callable

from splendor.config import EPSG, SEED
from splendor.loops import LoopData, get_loops
from splendor.landuse import N_TYPES

# Raster constants
GRID_SIZE = 100 # Pixels per side
PIXEL_M = 10    # Pixel size in meters


# Graphs
def _graph_with_attrs(G: nx.Graph, length_m: float = 1.) -> nx.Graph:
    """Add get_loops' expected edge_id and length_m attributes."""
    for eid, (u, v) in enumerate(G.edges()):
        G[u][v]["edge_id"] = eid
        G[u][v]["length_m"] = length_m
    return G


@pytest.fixture
def k3_graph() -> nx.Graph:
    """k3 with unit lengths."""
    return _graph_with_attrs(nx.complete_graph(3))


@pytest.fixture
def k3_loops(k3_graph) -> LoopData:
    """All loops of k3, with no length bound."""
    return get_loops(k3_graph, max_length_m=float("inf"))


@pytest.fixture
def k4_graph() -> nx.Graph:
    """k4 with unit lengths."""
    return _graph_with_attrs(nx.complete_graph(4))


@pytest.fixture
def k4_loops(k4_graph) -> LoopData:
    """All loops of k4, with no length bound."""
    return get_loops(k4_graph, max_length_m=float("inf"))


# Rasters / grids
def _grid_transform() -> Affine:
    """
    Transform for a GRID_SIZE x GRID_SIZE raster of PIXEL_M pixels,
    covering x, y in [0, GRID_SIZE * PIXEL_M].
    """
    return rasterio.transform.from_origin(
        west=0,                    # x of left edge
        north=GRID_SIZE * PIXEL_M, # y of top edge
        xsize=PIXEL_M,             # pixel width
        ysize=PIXEL_M              # pixel height
    )

@pytest.fixture
def half_raster() -> tuple[np.array, Affine]:
    """
    100x100 10m raster array with half forest values and half cropland values, 
    and Affine transform object.
    """
    d = 100
    raster = np.zeros((d, d), dtype=np.uint8)
    raster[:, :d//2] = 4 # Forest
    raster[:, d//2:] = 2 # Cropland
    return raster, _grid_transform()


@pytest.fixture
def random_raster() -> tuple[np.array, Affine]:
    """100x100 10m raster array with random values, and Affine transform object."""
    d = 100
    rng = np.random.default_rng(SEED)
    raster = rng.integers(0, N_TYPES + 1, size=(d, d), dtype=np.uint8)
    return raster, _grid_transform()


@pytest.fixture
def nodata_raster() -> tuple[np.array, Affine]:
    """100x100 10m raster array with all empty values, and Affine transform object."""
    d = 100
    raster = np.zeros((d, d), dtype=np.uint8)
    return raster, _grid_transform()


# Edges
@pytest.fixture
def make_edges() -> Callable:
    """Build an edges GeoDataFrame with edge_id 0..n-1 from list of coordinates."""
    def _make_edges(*coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"edge_id": range(len(coords))},
            geometry=[LineString(c) for c in coords],
            crs=EPSG
        )
    return _make_edges
