import numpy as np
from typing import Sequence, Tuple, Optional
import re

def gen_entity_array(n: int):
    """Generate an array of n entities named E0, E1, ... En-1."""
    return [f"E{i}" for i in range(n)]

def random_from_distribution(dist: str, low: float, high: float, precision: Optional[int] = None) -> float:
    """Generate a single random value"""
    if dist == "uniform":
        val = np.random.uniform(low, high)
    elif dist == "normal":
        mean = (low + high) / 2
        std = (high - low) / 6
        val = np.clip(np.random.normal(mean, std), low, high)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    if precision is not None and precision >= 0:
        s = str(val)
        # match digits before and after decimal up to precision places
        match = re.match(rf"(\d+\.)(\d+)", s)
        if match:
            s = f"{str(match.group(1))}{str(match.group(2))[:precision]}"  
            #print(s)     
             
        val = float(s)

    return val

def infer_precision(val: float) -> int:
    """Infer number of decimal places from a float as written."""
    s = f"{val}"
    if "e" in s or "E" in s:
        # handle scientific notation by expanding it
        s = f"{val:.16f}".rstrip("0")
    match = re.search(r"\.(\d+)", s)
    return len(match.group(1)) if match else 0

def minmax_scale_pairs(pairs: Sequence[Tuple[str, float]]):
    if not pairs:
        return []

    # extract numeric values
    values = np.array([v for _, v in pairs], dtype=float)
    mn, mx = values.min(), values.max()

    # infer average decimal precision across inputs
    precisions = [infer_precision(v) for _, v in pairs]
    avg_precision = int(round(np.mean(precisions))) if precisions else 0

    # normalize
    if mn == mx:
        normed = np.zeros_like(values)
    else:
        normed = (values - mn) / (mx - mn)

    # round normalized values to the same precision
    return [(e, round(float(v), avg_precision)) for (e, _), v in zip(pairs, normed)]

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
