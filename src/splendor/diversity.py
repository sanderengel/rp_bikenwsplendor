import numpy as np

from scipy.stats import entropy
from splendor.config import RICHNESS_MIN_SHARE


# Diversity measures
def _shannon(shares: np.ndarray) -> np.ndarray:
    return entropy(shares, axis=1)


def _inverse_simpson(shares: np.ndarray) -> np.ndarray:
    return 1 / (shares ** 2).sum(axis=1)


def _richness(shares: np.ndarray) -> np.ndarray:
    return (shares >= RICHNESS_MIN_SHARE).sum(axis=1)


MEASURES = {
    "shannon": _shannon,
    "inverse_simpson": _inverse_simpson,
    "richness": _richness
}


# Diversity measure caller
def get_diversity(shares: np.ndarray, measure_name: str) -> np.ndarray:
    if measure_name not in MEASURES:
        raise ValueError(f"Measure '{measure_name}' is not supported. Supported measures are {list(MEASURES.keys())}")
    row_sums = shares.sum(axis=1)
    if not np.allclose(row_sums, 1):
        raise ValueError("shares must sum to 1 per row")
    if (shares < 0).any():
        raise ValueError("shares must be non-negative")

    measure_func = MEASURES[measure_name]
    return measure_func(shares)
