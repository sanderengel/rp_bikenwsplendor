import math
import pytest
import numpy as np

from splendor import diversity
from splendor.diversity import get_diversity, MEASURES, get_bounds

DOMINATED = [.05, .05, .1, .8]
UNIFORM = [.25, .25, .25, .25]
HALF = [0, 0, .5, .5]
SINGLE = [0, 0, 0, 1]
NEAR_UNIFORM = [.15, .25, .35, .25]


@pytest.fixture(autouse=True)
def fixed_richness_threshold(monkeypatch):
    """Make richness tests independent of the config value."""
    monkeypatch.setattr(diversity, "RICHNESS_MIN_SHARE", 0.1)


@pytest.mark.parametrize("measure, shares, expected", [
    ("shannon", DOMINATED, 0.708347),
    ("shannon", UNIFORM, math.log(4)),
    ("shannon", HALF, math.log(2)),
    ("shannon", SINGLE, 0),
    ("shannon", NEAR_UNIFORM, 1.345153),
    ("inverse_simpson", DOMINATED, 1 / 0.655),
    ("inverse_simpson", UNIFORM, 4),
    ("inverse_simpson", HALF, 2),
    ("inverse_simpson", SINGLE, 1),
    ("richness", DOMINATED, 2),
    ("richness", UNIFORM, 4),
    ("richness", HALF, 2),
    ("richness", SINGLE, 1),
])
def test_diversity_values(measure, shares, expected):
    """Known values for single rows."""
    result = get_diversity(np.array([shares]), measure)
    assert result == pytest.approx([expected], abs=1e-6)


@pytest.mark.parametrize("measure", MEASURES)
def test_diversity_one_value_per_row(measure):
    """Each row gets its own value (guards against reducing over the wrong axis)."""
    shares = np.array([DOMINATED, UNIFORM, HALF, SINGLE, NEAR_UNIFORM])
    rowwise = np.concatenate([get_diversity(shares[[i]], measure) for i in range(len(shares))])
    result = get_diversity(shares, measure)

    assert result.shape == (len(shares),)
    np.testing.assert_allclose(result, rowwise)


@pytest.mark.parametrize("measure", MEASURES)
def test_diversity_ordering(measure):
    """Every measure ranks uniform >= dominated >= single type, and has no NaN on zero shares."""
    result = get_diversity(np.array([UNIFORM, DOMINATED, SINGLE]), measure)

    assert np.isfinite(result).all()
    assert result[0] >= result[1] >= result[2]


def test_diversity_unknown_measure_error():
    with pytest.raises(ValueError, match="not supported"):
        get_diversity(np.array([UNIFORM]), "unknown")


def test_diversity_rows_not_summing_to_one_error():
    with pytest.raises(ValueError, match="sum to 1"):
        get_diversity(np.array([[.5, .6]]), "shannon")


def test_diversity_negative_shares_error():
    with pytest.raises(ValueError, match="non-negative"):
        get_diversity(np.array([[1.5, -.5]]), "shannon") # Sums to 1


@pytest.mark.parametrize("measure, expected", [
    ("shannon", (0, math.log(13))),
    ("inverse_simpson", (1, 13)),
    ("richness", (1, 10)), # At most 10 types can reach a share of 0.1
])
def test_bounds(measure, expected):
    assert get_bounds(measure, 13, 0.1) == pytest.approx(expected)


def test_bounds_unknown_measure_error():
    with pytest.raises(ValueError, match="not supported"):
        get_bounds("unknown", 13, 0.1)
