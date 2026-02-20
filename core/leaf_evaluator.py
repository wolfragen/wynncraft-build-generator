import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def evaluate_leaf(
    ingredients,
    effectiveness,
    stat_ptr,
    stat_ids,
    stat_max,
    durability_arr,
    scaled_dura_max,
    relevant,
    has_min,
    has_max,
    min_vals,
    max_vals,
    weights,
    min_dura,
    dura_weight,
    LEFT,
    RIGHT,
    ABOVE,
    UNDER,
    TOUCHING,
    NOT_TOUCHING,
):
    # Allocate locally (Numba stack allocation, very cheap)
    slot_eff = np.zeros(6, dtype=np.int32)
    effective_stats = np.zeros(min_vals.shape[0], dtype=np.int32)

    # ---------------- EFFECTIVENESS ----------------

    for s in range(6):
        ing = ingredients[s]
        eff_row = effectiveness[ing]

        # LEFT
        t = LEFT[s]
        if t != -1:
            v = eff_row[0]
            if v != 0:
                slot_eff[t] += v

        # RIGHT
        t = RIGHT[s]
        if t != -1:
            v = eff_row[1]
            if v != 0:
                slot_eff[t] += v

        # ABOVE
        t = ABOVE[s]
        if t != -1:
            v = eff_row[2]
            if v != 0:
                slot_eff[t] += v

        # UNDER
        t = UNDER[s]
        if t != -1:
            v = eff_row[3]
            if v != 0:
                slot_eff[t] += v

        # TOUCHING
        v = eff_row[4]
        if v != 0:
            for j in range(TOUCHING.shape[1]):
                t = TOUCHING[s, j]
                if t != -1:
                    slot_eff[t] += v

        # NOT TOUCHING
        v = eff_row[5]
        if v != 0:
            for j in range(NOT_TOUCHING.shape[1]):
                t = NOT_TOUCHING[s, j]
                if t != -1:
                    slot_eff[t] += v

    # ---------------- STAT ACCUMULATION ----------------

    for s in range(6):
        ing = ingredients[s]
        mult = 100 + slot_eff[s]

        start = stat_ptr[ing]
        end = stat_ptr[ing + 1]

        for k in range(start, end):
            sid = stat_ids[k]
            raw = stat_max[k]
            effective_stats[sid] += (raw * mult) // 100

    # ---------------- DURABILITY ----------------

    durability = scaled_dura_max
    for s in range(6):
        durability += durability_arr[ingredients[s]]

    # ---------------- CONSTRAINT + SCORE ----------------

    score = 0.0

    for idx in relevant:
        val = effective_stats[idx]

        if has_min[idx] and val < min_vals[idx]:
            return -1.0

        if has_max[idx] and val > max_vals[idx]:
            return -1.0

        score += val * weights[idx]

    if min_dura != -1:
        if durability < min_dura:
            return -1.0

    if dura_weight != 0.0:
        score += durability * dura_weight

    return score