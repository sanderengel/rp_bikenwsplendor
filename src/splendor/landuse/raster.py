import math
import numpy as np
from rasterio.features import geometry_mask
from rasterio import windows
from shapely import Geometry
from affine import Affine


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