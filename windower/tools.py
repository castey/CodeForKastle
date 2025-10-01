import numpy as np
from typing import Sequence, Tuple, Optional

def gen_entity_array(n: int):
    """Generate an array of n entities named E0, E1, ... En-1."""
    return [f"E{i}" for i in range(n)]

def random_from_distribution(dist: str, low: float, high: float, precision: Optional[float] = None) -> float:
    """Return a random float in [low, high] given a distribution.

    Args:
        dist: Distribution type ("uniform" or "normal").
        low: Lower bound.
        high: Upper bound.
        precision: If given, round the output to the nearest multiple of this value.
                   For example, precision=1e-4 will round to 4 decimal places.
    """
    if dist == "uniform":
        val = np.random.uniform(low, high)
    elif dist == "normal":
        mean = (low + high) / 2
        std = (high - low) / 6
        val = np.clip(np.random.normal(mean, std), low, high)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    if precision:
        val = round(val / precision) * precision
    return val

def minmax_scale_pairs(pairs: Sequence[Tuple[str, float]]):
    if not pairs:
        return []
    values = np.array([v for _, v in pairs], dtype=float)
    mn, mx = values.min(), values.max()
    if mn == mx:
        return [(e, 0.0) for e, _ in pairs]
    normed = (values - mn) / (mx - mn)
    return [(e, float(v)) for (e, _), v in zip(pairs, normed)]

def split_pairs(pairs: Sequence[Tuple[str, float]]):
    mid = len(pairs) // 2
    return pairs[:mid], pairs[mid:]

def compute_less_than(pairs: Sequence[Tuple[str, float]]):
    result = {e: [] for e, _ in pairs}
    for e1, v1 in pairs:
        for e2, v2 in pairs:
            if v1 < v2:
                result[e1].append(e2)
    return result
