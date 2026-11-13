"""Aggregation of normalized components into a single score."""
from typing import Sequence
import numpy as np

TOL = 1e-9 # Tolerance for floating point values just outside [0, 1]


def _weighted_mean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compensatory: sum_i w_i * x_i. A high component can make up for a low one."""
    return matrix @ weights


def _weighted_geometric_mean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Non-compensatory: prod_i x_i ** w_i. A zero in any component with positive weight gives 0."""
    return np.prod(matrix ** weights, axis=1)


AGGREGATIONS = {
    "mean": _weighted_mean,
    "geometric": _weighted_geometric_mean
}


def aggregate(matrix: np.ndarray, method: str, weights: Sequence[float]) -> np.ndarray:
    """
    Aggregate normalized components (links x components, values in [0, 1]) into a single score.
    Weights are one per component, non-negative, and sum to 1.    
    """
    if method not in AGGREGATIONS:
        raise ValueError(f"Aggregation '{method}' is not supported. Supported aggregation methods are {list(AGGREGATIONS)}")

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D (links x components), got shape {matrix.shape}")
    if np.isnan(matrix).any():
        raise ValueError("matrix contains NaN")
    if (matrix < -TOL).any() or (matrix > 1 + TOL).any():
        raise ValueError(f"matrix must be in [0, 1]: range [{matrix.min():.4g}, {matrix.max():.4g}]")

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (matrix.shape[1],):
        raise ValueError(f"need one weight per component ({matrix.shape[1]}), got {weights.size}")
    if (weights < 0).any():
        raise ValueError(f"weights must be non-negative, got {weights}")
    if not np.isclose(weights.sum(), 1):
        raise ValueError(f"weights must sum to 1, got {weights.sum():.4g}")

    aggregate_func = AGGREGATIONS[method]
    return aggregate_func(matrix, weights)
