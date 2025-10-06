import numpy as np
from dist_analyzer import measure_distances

# ANSI colors
C_RESET   = "\033[0m"
C_HEADER  = "\033[96m"   # bright cyan
C_MIN     = "\033[92m"   # green
C_MAX     = "\033[91m"   # red
C_YELLOW  = "\033[93m"
C_MAGENTA = "\033[95m"
C_GRAY    = "\033[90m"

def print_run(label, result, all_results, color):
    """Pretty-print details for a single run with colors."""
    start, dists, path = result
    min_idx = dists.argmin()
    max_idx = dists.argmax()
    min_dist = dists[min_idx]
    max_dist = dists[max_idx]
    run_range = dists.max() - dists.min()

    print(f"{color}==== {label} ===={C_RESET}")
    print(f"{C_GRAY}Start index:{C_RESET} {start}")
    print(f"{C_GRAY}Path (first 10 indices):{C_RESET} {path[:10]} ...")
    print(f"{C_GRAY}Distances (first 10 shown):{C_RESET} {np.round(dists[:10], 4)}")
    print(f"{C_GRAY}Range:{C_RESET} {run_range:.4f}")
    print(f"{C_GRAY}Smallest distance:{C_RESET} {min_dist:.4f} "
          f"(between {path[min_idx]} → {path[min_idx+1]})")
    print(f"{C_GRAY}Largest distance:{C_RESET} {max_dist:.4f} "
          f"(between {path[max_idx]} → {path[max_idx+1]})")

    # Find all starts that yield this same range
    matching_starts = [s for s, r, _, _ in all_results if abs(r - run_range) < 1e-9]
    print(f"{C_MAGENTA}All starting indices yielding this same range:{C_RESET} {matching_starts}\n")

def run_distribution(name, embeddings):
    print(f"\n{C_HEADER}#################### {name} Distribution ####################{C_RESET}")
    print(f"{C_GRAY}Embeddings shape:{C_RESET} {embeddings.shape}")

    # Sweep all starts and collect (start, range, min_dist, max_dist)
    all_results = []
    for s in range(embeddings.shape[0]):
        dists, path = measure_distances(embeddings.copy(), starting_index=s)
        run_range = dists.max() - dists.min()
        all_results.append((s, run_range, dists, path))

    # Find best min and best max range runs
    best_min = min(all_results, key=lambda x: x[1])
    best_max = max(all_results, key=lambda x: x[1])

    # Print results
    print_run("Minimal Range Run", (best_min[0], best_min[2], best_min[3]), all_results, C_MIN)
    print_run("Maximal Range Run", (best_max[0], best_max[2], best_max[3]), all_results, C_MAX)

    # Collect unique min/max distances across all runs
    all_mins = [d.min() for _, _, d, _ in all_results]
    all_maxes = [d.max() for _, _, d, _ in all_results]
    unique_mins = np.unique(all_mins)
    unique_maxes = np.unique(all_maxes)

    print(f"{C_YELLOW}All unique min distances across ALL runs:{C_RESET} {np.round(unique_mins, 4)}")
    print(f"{C_YELLOW}All unique max distances across ALL runs:{C_RESET} {np.round(unique_maxes, 4)}")
    print(f"{C_HEADER}============================================================{C_RESET}")

# ----------------------
# Example usage
# ----------------------
def normalize(arr):
    """Scale array to [0,1]."""
    return (arr - arr.min()) / (arr.max() - arr.min())

if __name__ == "__main__":
    np.random.seed(42)
    n, dims = 100, 100

    # Uniform
    uniform_embeddings = np.random.rand(n, dims)

    # Normal -> normalize
    normal_embeddings = normalize(np.random.randn(n, dims))

    # Bimodal
    c1 = np.random.normal(loc=-5, scale=1, size=(n//2, dims))
    c2 = np.random.normal(loc=+5, scale=1, size=(n//2, dims))
    bimodal_embeddings = normalize(np.vstack([c1, c2]))

    # Septuple-modal
    clusters = []
    centers = np.linspace(-15, 15, 7)
    for c in centers:
        clusters.append(np.random.normal(loc=c, scale=1, size=(n//7, dims)))
    septuple_embeddings = normalize(np.vstack(clusters))

    # Run distributions
    run_distribution("Uniform", uniform_embeddings)
    run_distribution("Normal", normal_embeddings)
    run_distribution("Bimodal", bimodal_embeddings)
    run_distribution("Septuple-Modal", septuple_embeddings)
