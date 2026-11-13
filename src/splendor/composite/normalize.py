"""Normalization of score components to [0, 1]."""
import numpy as np
from scipy.stats import rankdata
from typing import Sequence

TOL = 1e-9 # Tolerance for floating point values just outside the bounds


def _scale_to_unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / (hi - lo), 0, 1)


def _by_bounds(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale by the theoretical range, i.e., (x - lo) / (hi - lo)."""
    if hi <= lo:
        raise ValueError(f"upper bound must be above lower bound, got [{lo}, {hi}]")
    if (values < lo - TOL).any() or (values > hi + TOL).any():
        raise ValueError(f"values outside bounds [{lo}, {hi}]: range [{values.min():.4g}, {values.max():.4g}]")
    return _scale_to_unit(values, lo, hi)


def _by_percentile(values: np.ndarray, lo: float, p: float) -> np.ndarray:
    """Scale by a high percentage of the data, capped at 1: (x - lo) / (P_p - lo)."""
    if not 0 < p <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    if (values < lo - TOL).any():
        raise ValueError(f"values below lower bound {lo}: minimum {values.min():.4g}")
    top = np.percentile(values, p)
    if top <= lo:
        raise ValueError(f"percentile {p} ({top:.4g}) is not above the lower bound {lo}")
    return _scale_to_unit(values, lo, top)


def _by_rank(values: np.ndarray) -> np.ndarray:
    """Percentile ranks: (r - 1) / (n - 1), with ties getting their minimum rank."""
    if values.size < 2:
        raise ValueError("rank normalization needs at least two values")
    return (rankdata(values, method="min") - 1) / (values.size - 1)


def normalize_all(
    values: np.ndarray,
    methods: Sequence[str],
    lo: float,
    hi: float,
    p: float,
) -> dict[str, np.ndarray]:
    """Normalize one component with each of the given methods, keyed by method name."""
    normalizers = {
        "bounds": lambda: _by_bounds(values, lo, hi),
        "percentile": lambda: _by_percentile(values, lo, p),
        "rank": lambda: _by_rank(values),
    }
    unknown = set(methods) - normalizers.keys()
    if unknown:
        raise ValueError(f"Normalizations {sorted(unknown)} are not supported. Supported are {list(normalizers)}")
    return {m: normalizers[m]() for m in methods}
