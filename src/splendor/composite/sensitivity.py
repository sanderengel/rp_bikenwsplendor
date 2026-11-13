"""Sensitivity of link scores to the methodological choices."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from itertools import product
from typing import Sequence

from splendor.composite.normalize import normalize_all
from splendor.composite.aggregate import aggregate

COMBINATION_COLUMNS = ["measure", "normalization", "aggregation", "weight"]


def get_score_grid(
    components: pd.DataFrame,
    measures: Sequence[str],
    normalizations: Sequence[str],
    aggregations: Sequence[str],
    land_use_weights: Sequence[float],
    bounds: dict[str, tuple[float, float]],
    p: float,
    geometric_floor: float,
    poi_column: str = "poi_diversity",
) -> pd.DataFrame:
    """
    Score every link under every combination of land use measure, normalization, aggregation and weight.
    'components' holds edge_id, one 'diversity_<measure>' column per measure and the POI column.
    'bounds' maps each component column to its (lo, hi). Each combination normalizes all
    components with the same method. Land use gets weight w, POI gets 1 - w.
    Returns long format: edge_id, measure, normalization, aggregation, weight, score, rank (1 = best).
    """
    edge_ids = components["edge_id"].to_numpy()

    poi = components[poi_column].to_numpy(dtype=float)
    poi_normalized = normalize_all(poi, normalizations, *bounds[poi_column], p)

    frames = []
    for measure in measures:
        land_use_column = f"diversity_{measure}"
        land_use = components[land_use_column].to_numpy(dtype=float)
        land_use_normalized = normalize_all(land_use, normalizations, *bounds[land_use_column], p)

        for normalization in normalizations:
            matrix = np.column_stack([land_use_normalized[normalization], poi_normalized[normalization]])

            for aggregation, w in product(aggregations, land_use_weights):
                floor = geometric_floor if aggregation == "geometric" else 0.0
                score = aggregate(matrix, aggregation, [w, 1 - w], floor)
                frames.append(pd.DataFrame({
                    "edge_id": edge_ids,
                    "measure": measure,
                    "normalization": normalization,
                    "aggregation": aggregation,
                    "weight": w,
                    "score": score,
                    "rank": rankdata(-score, method="min"), # Highest score gets rank 1
                }))

    return pd.concat(frames, ignore_index=True)


def summarize_ranks(grid: pd.DataFrame) -> pd.DataFrame:
    """Per link: median rank and rank spread over all combinations in the grid (1 = best)."""
    ranks = grid.groupby("edge_id")["rank"]
    summary = pd.DataFrame({
        "median_rank": ranks.median(),
        "q25_rank": ranks.quantile(0.25),
        "q75_rank": ranks.quantile(0.75),
        "min_rank": ranks.min(),
        "max_rank": ranks.max(),
    })
    summary["iqr_rank"] = summary["q75_rank"] - summary["q25_rank"]
    summary["range_rank"] = summary["max_rank"] - summary["min_rank"]
    return summary.reset_index()


def get_rank_correlations(grid: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the link scores of every pair of combinations."""
    wide = grid.pivot(index="edge_id", columns=COMBINATION_COLUMNS, values="score")
    return wide.corr(method="spearman")
