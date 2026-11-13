import math
import pytest
import numpy as np

from splendor.config import SEED
from splendor.composite.aggregate import aggregate, AGGREGATIONS

# Links A, B, C as (land use, POI)
MATRIX = np.array([
    [0.9, 0.0], # Diverse land without POIs
    [0.5, 2/3], # A bit of both
    [0.2, 1.0]  # All POIs on monotonous land
])
EQUAL = [0.5, 0.5]


def test_aggregate_mean():
    """Compensatory: C's full POI makes up for its monotonous land."""
    np.testing.assert_allclose(aggregate(MATRIX, "mean", EQUAL), [0.45, 0.25 + 1/3, 0.6])


def test_aggregate_geometric():
    """Non-compensatory: A has no POI and gets 0, and the balanced B scores highest."""
    np.testing.assert_allclose(
        aggregate(MATRIX, "geometric", EQUAL), [0, math.sqrt(0.5 * 2/3), math.sqrt(0.2)]
    )


@pytest.mark.parametrize("method", AGGREGATIONS)
def test_aggregate_all_weight_on_one_component(method):
    """With all weight on one component, the score is that component; weight 0 drops out (also as 0 ** 0)."""
    np.testing.assert_allclose(aggregate(MATRIX, method, [1, 0]), MATRIX[:, 0])
    np.testing.assert_allclose(aggregate(MATRIX, method, [0, 1]), MATRIX[:, 1])


@pytest.mark.parametrize("method", AGGREGATIONS)
def test_aggregate_in_unit_interval(method):
    """One score per link, in [0, 1]."""
    matrix = np.random.default_rng(SEED).uniform(0, 1, size=(50, 3))
    result = aggregate(matrix, method, [0.2, 0.3, 0.5])
    assert result.shape == (50,)
    assert ((result >= 0) & (result <= 1)).all()


def test_aggregate_unknown_method_error():
    with pytest.raises(ValueError, match="not supported"):
        aggregate(MATRIX, "unknown", EQUAL)


def test_aggregate_weights_not_summing_to_one_error():
    with pytest.raises(ValueError, match="sum to 1"):
        aggregate(MATRIX, "mean", [0.5, 0.6])


def test_aggregate_geometric_floor():
    """With a floor, a component at 0 is lifted to the floor instead of zeroing the geometric mean."""
    result = aggregate(MATRIX, "geometric", EQUAL, floor=0.25)
    expected = np.sqrt((0.25 + 0.75 * MATRIX[:, 0]) * (0.25 + 0.75 * MATRIX[:, 1]))
    np.testing.assert_allclose(result, expected)
    assert result[0] > 0 # Link A has no POI


def test_aggregate_floor_keeps_mean_ranking():
    """For the mean, the floor rescales the scores but doesn't change their order."""
    without_floor = aggregate(MATRIX, "mean", EQUAL)
    with_floor = aggregate(MATRIX, "mean", EQUAL, floor=0.25)
    np.testing.assert_array_equal(np.argsort(with_floor), np.argsort(without_floor))
