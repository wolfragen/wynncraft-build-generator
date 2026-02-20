import numpy as np

# 2 columns × 3 rows
# 0 1
# 2 3
# 4 5

LEFT  = np.array([-1, 0, -1, 2, -1, 4], dtype=np.int8)
RIGHT = np.array([1, -1, 3, -1, 5, -1], dtype=np.int8)
ABOVE = np.array([-1, -1, 0, 1, 2, 3], dtype=np.int8)
UNDER = np.array([2, 3, 4, 5, -1, -1], dtype=np.int8)

# Fixed-size padded arrays for Numba
TOUCHING = np.array([
    [1, 2, -1],
    [0, 3, -1],
    [0, 3, 4],
    [1, 2, 5],
    [2, 5, -1],
    [3, 4, -1],
], dtype=np.int8)

NOT_TOUCHING = np.array([
    [3, 4, 5],
    [2, 4, 5],
    [1, 5, -1],
    [0, 4, -1],
    [0, 1, 3],
    [0, 1, 2],
], dtype=np.int8)