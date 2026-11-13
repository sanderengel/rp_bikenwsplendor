import pytest
import numpy as np

from splendor.config import SEED
from splendor.composite.normalize import normalize_all

ALL_METHODS = ("bounds", "percentile", "rank")


def test_normalize_bounds():
    """Bounds map the theoretical range onto [0, 1]."""
    lo, hi = 0, 3
    p = 99
    result = normalize_all(np.array([0, 1.5, 3]), ["bounds"], lo, hi, p)
    np.testing.assert_allclose(result["bounds"], [0, 0.5, 1])


def test_normalize_bounds_floor():
    lo, hi = -1, 3
    p = 99
    """A lower bound below the data gives a floor: bounds (-1, 3) give (1 + POI) / 4."""
    result = normalize_all(np.array([0, 1, 2, 3]), ["bounds"], lo, hi, p)
    np.testing.assert_allclose(result["bounds"], [0.25, 0.5, 0.75, 1])


def test_normalize_percentile_caps_at_one():
    """Values are scaled by the percentile, and values above it are capped at 1."""
    lo, hi = 0, 100
    p = 50
    values = np.arange(101, dtype=float) # 0..100, 50th percentile is 50
    result = normalize_all(values, ["percentile"], lo, hi, p)["percentile"]
    assert result[25] == pytest.approx(0.5)
    assert (result[50:] == 1).all()


def test_normalize_rank():
    """Lowest value gets 0, highest gets 1."""
    lo, hi = 0, 3
    p = 99
    result = normalize_all(np.array([3., 1., 2.]), ["rank"], lo, hi, p)
    np.testing.assert_allclose(result["rank"], [1, 0, 0.5])


def test_normalize_rank_ties_get_minimum_rank():
    """Tied values share the lowest rank of their group."""
    lo, hi = 0, 3
    p = 99
    result = normalize_all(np.array([1., 1., 2.]), ["rank"], lo, hi, p)
    np.testing.assert_allclose(result["rank"], [0, 0, 1])


@pytest.mark.parametrize("method", ALL_METHODS)
def test_normalize_in_unit_interval(method):
    """Every method returns values in [0, 1]."""
    lo, hi = 1, 5
    p = 90
    values = np.random.default_rng(SEED).uniform(lo, hi, size=50)
    result = normalize_all(values, [method], lo, hi, p)[method]
    assert ((result >= 0) & (result <= 1)).all()


def test_normalize_unknown_method_error():
    lo, hi = 0, 1
    p = 99
    with pytest.raises(ValueError, match="not supported"):
        normalize_all(np.array([0., 1.]), ["unknown"], lo, hi, p)
