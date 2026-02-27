import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def evaluate_leaf(
    ingredients,
    k,
    stat_min_matrix,
    stat_max_matrix,
    base_min,
    base_max,
    void_effectiveness,
    use_eff,
    durability_idx,
    has_min_mask,
    has_max_mask,
    min_vals,
    max_vals,
    weights,
):

    K = min_vals.shape[0]

    acc_min = np.empty(K, dtype=np.int32)
    acc_max = np.empty(K, dtype=np.int32)

    # ---------------- INIT ----------------
    for s in range(K):
        acc_min[s] = base_min[s]
        acc_max[s] = base_max[s]

    # ---------------- ACCUMULATION ----------------
    for i in range(k):

        ing = ingredients[i]

        mult = void_effectiveness[i] if use_eff else 100

        row_min = stat_min_matrix[ing]
        row_max = stat_max_matrix[ing]

        for s in range(K):

            if s == durability_idx:
                acc_min[s] += row_min[s]
                acc_max[s] += row_max[s]
            else:
                acc_min[s] += (row_min[s] * mult) // 100
                acc_max[s] += (row_max[s] * mult) // 100

    # ---------------- CONSTRAINT + SCORE ----------------
    score = 0.0

    for s in range(K):

        min_v = acc_min[s]
        max_v = acc_max[s]

        if has_min_mask[s] and max_v < min_vals[s]:
            return -np.inf, acc_min, acc_max

        if has_max_mask[s] and min_v > max_vals[s]:
            return -np.inf, acc_min, acc_max

        score += max_v * weights[s] * 0.9
        score += min_v * weights[s] * 0.1

    return score, acc_min, acc_max