"""
Loop-level attributes computed from edge attributes.
"""
import numpy as np
from splendor.loops import LoopData


def loop_sum(loops: LoopData, edge_values: np.ndarray) -> np.ndarray:
    """Sum of an edge attribute over each loop's edges."""
    return loops.M.astype(np.float64) @ edge_values


def loop_max(loops: LoopData, edge_values: np.ndarray) -> np.ndarray:
    """Maximum of an edge attribute over each loop's edges."""
    if loops.n_loops == 0:
        return np.empty(0, dtype=edge_values.dtype)
    return np.maximum.reduceat(edge_values[loops.edges], loops.ptr[:-1])


def loop_any(loops: LoopData, edge_flags: np.ndarray) -> np.ndarray:
    """True for loops where at least one edge has the flag set."""
    return (loops.M @ edge_flags.astype(np.int32)) > 0


def loop_poi_diversity(
    loops: LoopData,
    has_facility: np.ndarray,
    has_service: np.ndarray,
    has_attraction: np.ndarray,
) -> np.ndarray:
    """Number of POI categories present on each loop."""
    return (
        loop_any(loops, has_facility).astype(np.int8)
        + loop_any(loops, has_service)
        + loop_any(loops, has_attraction)
    )
