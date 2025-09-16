import numpy as np
from typing import Sequence, Optional, Tuple

class Window:
    """
    A window node that contains a range of elements and links to left and right children nodes.
    
    Attributes:
        -elements (list[float]): The numeric values stored at this node. Intended
            to be pre-scaled to [0, 1], but not enforced.
            
        -n (int): Remaining depth for this branch (1 = leaf).
        
        -left (Optional[Window]): Left child (first half), or None for leaves.

        -right (Optional[Window]): Right child (second half), or None for leaves.
    """
    
    def __init__(self, elements: Sequence[float], n: int):
        """Initialize a Window.

        Args:
            elements: Values to store at this node.
            n: Remaining depth to recurse when building the tree.
        """
        
        self.elements = list(elements)   # assume already normalized to [0,1]
        self.n = n
        
        # window nodes contain optional child windows divided into left and right
        self.left: Optional["Window"] = None
        self.right: Optional["Window"] = None

    # this might be useful 
    def __repr__(self):
        return f"Window(n={self.n}, len={len(self.elements)})"

# min-max normalization function
def minmax_scale(arr: Sequence[float]) -> np.ndarray:
    """
    Args:
        arr: 1-D sequence of numeric values.

    Returns:
        np.ndarray: Array of floats scaled to [0, 1].
    """
    
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# function to split sequence (array) in half for creating child windows
def split_sequence(arr: Sequence[float]):
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]

def windower(array: Sequence[float], n: int = 1) -> Window:
    """Recursively build a binary tree of windows by halving the input each level.

    At each node, the current slice of `array` is stored. Recursion stops when
    `n <= 1` or the slice length is <= 1.

    Args:
        array: Values for this branch (ideally already scaled to [0, 1]).
        n: Depth of the window tree to build (>= 1).

    Returns:
        Window: Root node of the constructed window tree.
    """

    # create top level node
    node = Window(array, n)
    
    # check for depth
    if n <= 1 or len(array) <= 1:
        return node
    
    # split array into left and right arrays 
    left, right = split_sequence(array)
    
    # recursively run windower on child nodes
    node.left = windower(left, n - 1)
    node.right = windower(right, n - 1)
    
    return node

# ==========================

# ==== pretty printing ====
def _round_list(xs, ndigits_rounded=3):
    # ensure pure Python floats + rounding
    return [round(float(x), ndigits_rounded) for x in xs]

def print_windows(node: Window, path="root", ndigits_rounded=3):
    if node is None:
        return
    elems = _round_list(node.elements, ndigits_rounded)
    print(f"Window depth={node.n}, path={path}, elements={elems}")
    if node.left:
        print_windows(node.left, path + "->L", ndigits_rounded)
    if node.right:
        print_windows(node.right, path + "->R", ndigits_rounded)
# =========================

def run_windower_pipeline(
    values: Sequence[float], # input array 
    *, 
    depth: int = 5, # windower depth
    sort_input: bool = False, # sort array before processing?
    verbose: bool = True,
    ndigits_rounded: int = 3, # rounding for pretty printing
) -> Tuple[np.ndarray, Window]:
    """End-to-end pipeline: normalize values, build window tree, and optionally print.

    Args:
        -values: Any 1-D numeric sequence (list/tuple/np.ndarray).
        -depth: Target depth of the window tree (>= 1).
        -sort_input: If True, sort values ascending before normalization/windowing.
        -verbose: If True, print a small report including the tree.
        -ndigits_rounded: Decimal places to display when pretty printing.

    Returns:
        Tuple[np.ndarray, Window]:
            - The normalized NumPy array scaled to [0, 1].
            - The root Window of the constructed tree.
    """
    
    # convert to numpy array 
    arr = np.asarray(values, dtype=float)

    if sort_input:
        arr = np.sort(arr)

    # min-max scaling
    prepared = minmax_scale(arr)

    # run windower and set to variable containing root node
    root = windower(prepared, depth)

    # computer yap
    if verbose:
        
        # print raw 
        print("\nPre-normalized input")
        print(_round_list(values, ndigits_rounded))
        
        # print normal
        print("Input (normalized to [0, 1]):")
        print(_round_list(prepared, ndigits_rounded))
        
        # print windows with depth and path
        print("\nWindows (by depth & path):")
        print_windows(root, "root", ndigits_rounded)

    # return the scaled array and the window tree
    return prepared, root

if __name__ == "__main__":

    rng = np.random.default_rng(123)
    raw = rng.normal(loc=5.0, scale=2.0, size=10)  # arbitrary distribution

    norm, root = run_windower_pipeline(
        raw,
        depth=4,
        sort_input=True,   # keep ascending order
        verbose=True,
        ndigits_rounded=3,
    )

