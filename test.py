# This script replaces the old __main__ block. It imports your module
# and runs the same demonstration steps.

import numpy as np
from windower import minmax_scale, windower, print_windows, _round_list

def main() -> None:
    # config
    size = 10        # number of values
    low, high = 0, 67
    depth = 5
    ndigits = 3      # rounding for display

    # 1) original array
    arr = np.random.uniform(low, high, size)
    arr.sort()
    print("Original array:")
    print(_round_list(arr, ndigits))

    # 2) normalized array
    norm = minmax_scale(arr)
    print("\nNormalized array [0,1]:")
    print(_round_list(norm, ndigits))

    # 3) build windows
    root = windower(norm, depth)

    print("\nWindows (by depth & path):")
    print_windows(root, "root", ndigits)

if __name__ == "__main__":
    main()
