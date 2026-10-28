import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rasterio import windows
from shapely import Geometry
from shapely.geometry import box
from affine import Affine
from typing import Sequence

from splendor.landuse import ALL_TYPES, N_TYPES, SHORT_NAMES


def _landuse_colormap() -> tuple[list[tuple[float, ...]], ListedColormap, BoundaryNorm]:
    """
    Colors, colormap and norm for land use type ids.
    Index k = type id k (no data is transparent).
    """
    colors = [(1, 1, 1, 0)] + [tuple(c / 255 for c in rgb) for _, _, _, rgb in ALL_TYPES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, N_TYPES + 1), cmap.N)
    return colors, cmap, norm


def get_map_extent(geom: Geometry, margin: float) -> tuple[float, float, float, float]:
    """Square extent around a geometry padded by 'margin' of its larger side."""
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) * (.5 + margin)
    return cx - half, cy - half, cx + half, cy + half


def plot_landuse_map(
    ax: Axes,
    raster: np.ndarray,
    transform: Affine,
    line: Geometry,
    corridor: Geometry,
    margin: float = .15,
) -> None:
    """Plot land use around one link on the given ax."""
    _, cmap, norm = _landuse_colormap()

    # Raster window around the corridor
    minx, miny, maxx, maxy = get_map_extent(corridor, margin)
    window = windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    window = window.round_offsets().round_lengths()
    left, bottom, right, top = windows.bounds(window, transform)
    ax.imshow(
        raster[window.toslices()], cmap=cmap, norm=norm, 
        extent=(left, right, bottom, top), interpolation="nearest"
    )

    # Dim everything outside the corridor
    outside = box(left, bottom, right, top).difference(corridor)
    gpd.GeoSeries([outside]).plot(ax=ax, color="white", alpha=.65, linewidth=0)
    gpd.GeoSeries([corridor]).boundary.plot(ax=ax, color="black", linewidth=.6, linestyle="--")
    ax.plot(*line.xy, color="black", linewidth=1.5)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_shares(
    ax: Axes,
    row: pd.Series,
    min_share: float = .01,
    box_aspect: float = 1.6,
) -> None:
    """Horizontal bars of the link's land use shares in fixed type order."""
    colors, _, _ = _landuse_colormap()

    # Extract shares above minimum for the link
    s = pd.Series({k: row[f"share_{k}"] for k in range(1, N_TYPES + 1)})
    s = s[s >= min_share]

    # Create rows
    n_rows = max(len(s), 5) # Fixed minimum
    y = n_rows - 1 - np.arange(len(s)) # First type at the top
    ax.barh(y, s.to_numpy(), color=[colors[k] for k in s.index], height=.7)
    ax.set_yticks(y, [SHORT_NAMES[k] for k in s.index], fontsize=8)
    for yi, v in zip(y, s.to_numpy()):
        ax.text(v + .03, yi, f"{v:.0%}", va="center", fontsize=8, color="0.3")

    ax.set_xlim(0, 1.3)
    ax.set_ylim(-.6, n_rows - .4)
    ax.set_box_aspect(box_aspect) # Same height as square map next to it
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def plot_link_landuse_grid(
    edge_ids: Sequence[int],
    edges: gpd.GeoDataFrame,
    corridors: gpd.GeoSeries,
    shares: pd.DataFrame,
    raster: np.ndarray,
    transform: Affine,
    n_cols: int = 4,
    title: str | None = None,
    min_share: float = .01,
    margin: float = .15,
    map_width_ratio: float = 1.6,
    col_width: float = 3.8,
    row_height: float = 2.5,
    title_height: float = .3,
) -> Figure:
    """Grid of panels, one per link: land use map with corridor, and its share bars."""
    n_rows = -(-len(edge_ids) // n_cols)
    fig = plt.figure(figsize=(col_width * n_cols, row_height * n_rows), layout="constrained")
    title_fig, grid_fig = fig.subfigures(2, 1, height_ratios=[title_height, row_height * n_rows])
    subfigs = grid_fig.subfigures(n_rows, n_cols, wspace=.03, hspace=0, squeeze=False)

    for subfig, edge_id in zip(subfigs.flat, edge_ids):
        edge_id = int(edge_id)
        line = edges.geometry.iloc[edge_id]
        ax_map, ax_bar = subfig.subplots(1, 2, width_ratios=[map_width_ratio, 1])

        plot_landuse_map(ax_map, raster, transform, line, corridors.iloc[edge_id], margin)
        n_pixels = int(shares.at[edge_id, "n_pixels"])
        ax_map.set_title(
            f"Link {edge_id} ({line.length / 1000:.1f} km, {n_pixels:,} pixels)",
            fontsize=10, loc="left"
        )

        plot_shares(ax_bar, shares.loc[edge_id], min_share, map_width_ratio)

    if title:
        title_fig.suptitle(title, fontsize=14)
    return fig
