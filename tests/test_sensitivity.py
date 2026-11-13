import math
import pytest
import numpy as np
import pandas as pd

from splendor.composite.sensitivity import (
    get_score_grid,
    summarize_ranks,
    get_rank_correlations,
    COMBINATION_COLUMNS,
)

MEASURES = ("shannon", "richness")
NORMALIZATIONS = ("bounds", "percentile", "rank")
AGGREGATIONS = ("mean", "geometric")
WEIGHTS = (0.5, 1.0)
P = 99
N_COMBINATIONS = len(MEASURES) * len(NORMALIZATIONS) * len(AGGREGATIONS) * len(WEIGHTS)

BOUNDS = {
    "diversity_shannon": (0, math.log(13)),
    "diversity_richness": (1, 10),
    "poi_diversity": (0, 3),
}


@pytest.fixture
def components() -> pd.DataFrame:
    """Four links with non-consecutive edge ids."""
    return pd.DataFrame({
        "edge_id": [10, 11, 12, 13],
        "diversity_shannon": [0.2, 0.8, 1.2, 0.5],
        "diversity_richness": [1, 2, 3, 2],
        "poi_diversity": [0, 1, 3, 2],
    })


@pytest.fixture
def grid(components) -> pd.DataFrame:
    return get_score_grid(components, MEASURES, NORMALIZATIONS, AGGREGATIONS, WEIGHTS, BOUNDS, P, 0.25)


def test_grid_covers_every_combination_for_every_link(grid, components):
    sizes = grid.groupby(COMBINATION_COLUMNS).size()
    assert len(sizes) == N_COMBINATIONS
    assert (sizes == len(components)).all()


def test_grid_keeps_edge_ids(grid, components):
    """Every combination has all links, in the input order."""
    for _, group in grid.groupby(COMBINATION_COLUMNS):
        assert group["edge_id"].to_list() == components["edge_id"].to_list()


def test_grid_known_score(grid, components):
    """Shannon, bounds, mean, weight 1: the score is Shannon / ln 13, and the highest Shannon gets rank 1."""
    combination = grid.query(
        "measure == 'shannon' and normalization == 'bounds' and aggregation == 'mean' and weight == 1.0"
    )
    expected = components["diversity_shannon"].to_numpy() / math.log(13)
    np.testing.assert_allclose(combination["score"], expected)
    assert combination.set_index("edge_id").loc[12, "rank"] == 1


def test_summary_one_row_per_link(grid, components):
    summary = summarize_ranks(grid)
    assert summary["edge_id"].to_list() == sorted(components["edge_id"])


def test_rank_correlations_shape_and_diagonal(grid):
    """One row and column per combination, and every combination correlates perfectly with itself."""
    correlations = get_rank_correlations(grid)
    assert correlations.shape == (N_COMBINATIONS, N_COMBINATIONS)
    np.testing.assert_allclose(np.diag(correlations.to_numpy()), 1)
