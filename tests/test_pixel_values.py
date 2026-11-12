import numpy as np
from rasterio.features import geometry_mask
from shapely import LineString

from splendor.landuse.corridors import get_link_corridor
from splendor.landuse.raster import get_pixel_values


def test_pixel_values_match_full_mask(random_raster):
    """
    Pixels found through the window should be exactly those whose center lies in the geometry,
    also when the geometry's bounds are not on pixel edges.
    """
    x, y = 123.4, 217.9
    xL, yL = 437.9, 470.3
    r = 37
    line = LineString([(x, y), (x + xL, y + yL)]) # Off the pixel grid
    corridor = get_link_corridor(line, r)

    raster, transform = random_raster
    values = get_pixel_values(corridor, raster, transform)

    # Brute force, mask over the whole raster
    full_mask = geometry_mask([corridor], out_shape=raster.shape, transform=transform, invert=True)
    np.testing.assert_array_equal(np.sort(values), np.sort(raster[full_mask]))
