"""
The purpose of this script is to measure the spread of vectors in a high 
dimensional embeddng space. We do this by picking an arbitrary vector at random
and finding the nearest vector to it and logging the distance. We then measure 
from that second vector to the nearest vector that has not been visited. 

After logging all these distances, we sort and plot them. 

For evenly distributed datasets, we expect to find a low range of distances.
For highly clustered datasets with multiple distinct clusers, we expect
a high range of distances.  

"""
import numpy as np

def measure_distances(embeddings: np.ndarray, starting_index: int):
    """
    Greedy nearest-neighbor walk through an embedding space.
    Start at a random vector, hop to nearest unvisited neighbor each step,
    record the distances.
    
    Mutates `embeddings` by marking visited rows as np.inf.
    """
    
    if embeddings.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array (n, d)!")
    
    num_of_vectors = embeddings.shape[0]
    
    # pick a random starting index
    if starting_index is not None:
        if starting_index > num_of_vectors - 1:
            raise ValueError("Starting index out of bounds!")
        current = starting_index
        
    else:    
        current = np.random.randint(0, num_of_vectors)
    
    path = [ current ]
    distances = []
    
    for _ in range(num_of_vectors - 1):
        # distances from current to all vectors
        dists = np.linalg.norm(embeddings - embeddings[current], axis=1)
        
        # mark the current vector as visited
        dists[current] = np.inf
        embeddings[current] = np.inf
        
        # nearest unvisited neighbor
        next_index = np.argmin(dists)
        
        distances.append(dists[next_index])
        path.append(next_index)
        current = next_index
    
    return np.array(distances), np.array(path)

def find_extreme_ranges(embeddings: np.ndarray):
    """
    Run measure_distances from every possible starting index.
    Return the results (distances and paths) for the run with the minimal range
    and the run with the maximal range.
    """
    n = embeddings.shape[0]
    min_range = float("inf")
    max_range = float("-inf")
    best_min = None
    best_max = None

    for start_idx in range(n):
        dists, path = measure_distances(embeddings.copy(), starting_index=start_idx)
        d_range = dists.max() - dists.min()

        if d_range < min_range:
            min_range = d_range
            best_min = (start_idx, dists, path)

        if d_range > max_range:
            max_range = d_range
            best_max = (start_idx, dists, path)

    return best_min, best_max



    
    
