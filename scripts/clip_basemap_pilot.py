"""
Clip Basemap05 (lu_01, 2024) to the pilot area and make a styled main-type map.

Writes to data/interim/pilot/:
- basemap05_lu01.tif: original pixel values, clipped (for analysis)
- basemap05_main_types.tif: pixels grouped into main ecosystem types, with colours (for viewing)
- basemap05_main_types.csv: legend
"""
import numpy as np
from pathlib import Path
import pyogrio
import rasterio
from rasterio.windows import from_bounds

from splendor.io import load_network
from splendor.landuse import MAIN_TYPES, UNCLASSIFIED

RAW = Path("data/raw/basemap_05_public_geotif")
SRC = RAW / "lu_01_2024.tif"
VAT = RAW / "lu_01_2024.tif.vat.dbf"
NETWORK = Path("data/interim/pilot/network.gpkg")
OUT = Path("data/interim/pilot")

OUT.mkdir(parents=True, exist_ok=True)

BUFFER_M = 1000 # Margin around the network's bounding box


# Value: main type lookup from the raster attribute table
vat = pyogrio.read_dataframe(VAT)
prefix = (vat["C_26"].astype(int) // 1000).to_numpy() # E.g., 104200 -> 104
# Position v holds the main-type id for pixel value v
lookup = np.zeros(int(vat["Value"].max()) + 1, dtype=np.uint8)
for value, p in zip(vat["Value"].to_numpy(), prefix):
    lookup[value] = MAIN_TYPES[p][0] if p in MAIN_TYPES else UNCLASSIFIED[0]


# Clip window around the network
_, edges = load_network(NETWORK)
minx, miny, maxx, maxy = edges.total_bounds
bounds = (minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M)

with rasterio.open(SRC) as src:
    print(f"Basemap05 CRS: {src.crs}, network CRS: {edges.crs}")

    # Define window, src.transform maps pixel positions to map coordinates
    window = from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()

    # 2D array of pixel values inside the window
    data = src.read(1, window=window)

    # Settings for the new GeoTIFF
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": data.dtype,
        "crs": src.crs,
        "transform": src.window_transform(window),
        "height": data.shape[0],
        "width": data.shape[1],
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    }

# Clipped original values
with rasterio.open(OUT / "basemap05_lu01.tif", "w", **profile) as dst:
    dst.write(data, 1)

# Main types with a color table (shown automatically in QGIS)
main = lookup[data]
with rasterio.open(OUT / "basemap05_main_types.tif", "w", **(profile | {"dtype": "uint8"})) as dst:
    dst.write(main, 1)
    dst.write_colormap(1, {i: (*rgb, 255) for i, _, rgb in [*MAIN_TYPES.values(), UNCLASSIFIED]})

# Create legend txt file for QGIS
with open(OUT / "basemap05_main_types.txt", "w") as f:
    f.write("# QGIS Generated Color Map Export File\nINTERPOLATION:EXACT\n")
    for i, name, (r, g, b) in [*MAIN_TYPES.values(), UNCLASSIFIED]:
        f.write(f"{i},{r},{g},{b},255,{name}\n")

# Print percentages
counts = np.bincount(main.ravel(), minlength=14)
print(f"Clipped {data.shape[1]}x{data.shape[0]} pixels")
for i, name, _ in [*MAIN_TYPES.values(), UNCLASSIFIED]:
    print(f"{i:2d} {name:35s} {counts[i] / counts[1:].sum():6.1%}")
