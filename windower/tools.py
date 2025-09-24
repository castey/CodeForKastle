import numpy as np

def gen_entity_array(n: int):
    """Generate an array of n entities named E0, E1, ... En-1."""
    return [f"E{i}" for i in range(n)]

def random_from_distribution(dist: str, low: float, high: float) -> float:
    """Return a random float in [low, high] given a distribution."""
    if dist == "uniform":
        return np.random.uniform(low, high)
    elif dist == "normal":
        mean = (low + high) / 2
        std = (high - low) / 6
        return np.clip(np.random.normal(mean, std), low, high)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
