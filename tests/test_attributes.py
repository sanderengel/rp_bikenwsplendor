import numpy as np
from splendor.attributes import loop_sum, loop_max, loop_poi_diversity


def test_loop_sum_equals_length(k4_loops):
    """loop_sum of lengths should equal loops.lengths_m."""
    np.testing.assert_array_equal(loop_sum(k4_loops, np.ones(6)), k4_loops.lengths_m)


def test_loop_max_one_steep_edge(k4_loops):
    """loop_max of max_slope should give the correct slope with a single steep edge."""
    slopes = np.ones(6)
    slopes[0] = 5.0
    max_slope = loop_max(k4_loops, slopes)
    assert (max_slope == 5.0).sum() == 4
    assert (max_slope == 1.0).sum() == 3


def test_poi_diversity(k4_loops):
    """Facility only on one edge, attraction everywhere, no services."""
    has_facility = np.zeros(6)
    has_facility[0] = 1
    has_service = np.zeros(6)
    has_attraction = np.ones(6)
    poi_diversity = loop_poi_diversity(k4_loops, has_facility, has_service, has_attraction)
    # Four loops contain edge 0 (facility + attraction), three don't (only attraction)
    assert sorted(poi_diversity.tolist()) == [1, 1, 1, 2, 2, 2, 2]
