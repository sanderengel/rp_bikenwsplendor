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
        raise ValueError(f"Measure '{measure_name}' is not supported. Supported measures are {list(MEASURES)}")
    row_sums = shares.sum(axis=1)
    if not np.allclose(row_sums, 1):
        raise ValueError("shares must sum to 1 per row")
    if (shares < 0).any():
        raise ValueError("shares must be non-negative")

    measure_func = MEASURES[measure_name]
    return measure_func(shares)


def get_bounds(measure_name: str, n_types: int, richness_min_share: float) -> tuple[float, float]:
    """Theoretical (min, max) of a diversity measure over n_types land use types."""
    if measure_name == "shannon":
        return 0.0, float(np.log(n_types))
    if measure_name == "inverse_simpson":
        return 1.0, float(n_types)
    if measure_name == "richness":
        return 1.0, float(min(n_types, int(1 / richness_min_share))) # At most 1 / threshold types can reach this
    raise ValueError(f"Measure '{measure_name}' is not supported. Supported measures are {list(MEASURES)}")
