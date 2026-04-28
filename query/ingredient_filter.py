"""
ingredient_filter.py

Filters raw ingredients according to Query rules.

Exact same logic as original implementation,
adapted to dense stat vectors.

Effectiveness filtering removed.
"""

from data.stats import STAT_INDEX, IDX_DURABILITY, IDX_DURATION, IDX_CHARGES, REQ_STATS_IDX
from data.ingredient_loader import SKILL_INDEX

import numpy as np
from numba import njit


def filter_raw_ingredients(
    ingredients_raw,
    query,
    recipe,
):
    """
    Takes all ingredients and the user Query, then returns only useful ingredients.
    """

    filtered = []

    has_min = query.has_min_mask
    has_max = query.has_max_mask
    weights = query.weights
    search_inv = query.search_for_inversion
    skill = query.skill
    filter_mask = query.filter_mask
    stat_index = STAT_INDEX

    # Pre-resolve skill index if needed
    if skill is not None:
        skill_index = SKILL_INDEX.get(skill)
        if skill_index is None:
            return []
    else:
        skill_index = None

    for ing in ingredients_raw:

        # ---- Skill filter ----
        if skill_index is not None:
            if not ing.skills[skill_index]:
                continue
        
        # ---- posMods filter ----
        if any(x != 0 for x in ing.pos_mods):
            continue

        keep = False

        min_stats = ing.stats_min
        max_stats = ing.stats_max
    
        if not query.consumable :
            # ---- positive durability filter ----
            if max_stats[IDX_DURABILITY] > 0:
                continue
        else:
            # ---- positive duration filter ----
            if max_stats[IDX_DURATION] > 0:
                continue
            if max_stats[IDX_CHARGES] != 0:
                continue

        # ---- Normal stats ----
        for stat_name, idx in stat_index.items():

            if idx == IDX_DURABILITY or idx == IDX_DURATION:
                continue

            # Skip stats not used in query
            if not (has_min[idx] or has_max[idx] or weights[idx] != 0):
                continue

            # skip stats opted out of ingredient filtering
            if not filter_mask[idx]:
                continue

            min_val = min_stats[idx]
            max_val = max_stats[idx]

            # ---- Minimum defined and stat can be positive ----
            if has_min[idx] and max_val > 0:
                keep = True
                break

            # ---- Maximum defined and stat can be negative ----
            if has_max[idx] and min_val < 0:
                keep = True
                break

            # ---- Positive weight defined and can be positive ----
            if weights[idx] > 0 and max_val > 0:
                keep = True
                break

            # ---- Negative weight defined and can be negative ----
            if weights[idx] < 0 and min_val < 0:
                keep = True
                break

            # ---- Inversion if enabled ----
            if search_inv:

                if has_min[idx] and min_val < 0:
                    keep = True
                    break

                if has_max[idx] and max_val > 0:
                    keep = True
                    break

                if weights[idx] > 0 and min_val < 0:
                    keep = True
                    break

                if weights[idx] < 0 and max_val > 0:
                    keep = True
                    break

        if keep:
            filtered.append(ing)
    n_filtered = len(filtered)
    filtered = pareto_cull_ingredients(filtered, query, recipe)
    n_culled = len(filtered)
    
    pct = (n_culled / n_filtered * 100) if n_filtered else 0.0
    print(f"Ingredient culling: {n_filtered} => {n_culled}, {pct:.2f}% left")
    return filtered


def pareto_cull_ingredients(ingredients, query, recipe):

    if len(ingredients) <= 1:
        return ingredients

    active = query.active_indices
    req_idx = query.req_idx
    stat_count = query.stat_count
    dura_proj_idx = query.dura_proj_idx

    dur_idx = IDX_DURATION if query.consumable else IDX_DURABILITY

    matrix = np.zeros((len(ingredients), stat_count), dtype=np.int32)

    for i, ing in enumerate(ingredients):

        for j, stat_idx in enumerate(active):

            min_val = ing.stats_min[stat_idx]
            max_val = ing.stats_max[stat_idx]

            # ---- apply recipe dura shift ----
            if stat_idx == dur_idx:
                min_val += recipe.scaled_dura_min
                max_val += recipe.scaled_dura_max

            # ---- select best representative value ----
            if stat_idx in REQ_STATS_IDX:
                if max_val > 0:
                    matrix[i, j] = min_val
                else:
                    matrix[i, j] = max_val
            elif max_val > 0:
                matrix[i, j] = max_val
            else:
                matrix[i, j] = min_val

    kept_mask = pareto_filter_ingredients(matrix, req_idx, dura_proj_idx)

    return [ingredients[i] for i in range(len(ingredients)) if kept_mask[i]]


@njit(cache=True)
def pareto_filter_ingredients(matrix, req_idx, dura_proj_idx):

    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    for i in range(n):

        if not is_kept[i]:
            continue

        for j in range(i + 1, n):

            if not is_kept[j]:
                continue

            cmp = compare_stat_vectors(matrix[i], matrix[j], req_idx, dura_proj_idx)

            if cmp == 1:
                is_kept[j] = False

            elif cmp == -1:
                is_kept[i] = False
                break

            elif cmp == 2:
                is_kept[j] = False

    return is_kept


@njit(cache=True)
def compare_stat_vectors(a, b, req_idx, dura_i):
    """
    Returns:
        1  if A dominates B
       -1  if B dominates A
        2  if identical
        0  if incomparable

    `dura_i` is the column of durability/duration (or -1). Dura is
    non-invertible, so it follows a strict "higher is better" rule and
    ignores the sign-mismatch incomparability guard.
    """

    a_better = False
    b_better = False

    for i in range(len(a)):

        va = a[i]
        vb = b[i]

        if va == vb:
            continue

        # Dura/duration: strictly higher = better, fully comparable.
        if i == dura_i:
            if va > vb: a_better = True
            else:       b_better = True
            if a_better and b_better:
                return 0
            continue

        # Strict sign opposition → incomparable.
        if (va > 0 and vb < 0) or (va < 0 and vb > 0):
            return 0

        is_req = i in req_idx

        if va > 0 or vb > 0:
            # Both >= 0 (with at least one > 0).
            # For reqs, lower is better → flip direction.
            if (va > vb) != is_req:
                a_better = True
            else:
                b_better = True
        else:
            # Both <= 0.
            # For non-req invertible stats, "0 vs -X" is incomparable:
            # the 0 side has no stat at all, while -X is only useful via
            # -eff inversion for one direction but not the other.
            if not is_req and (va == 0 or vb == 0):
                return 0
            # Otherwise: further-below-zero = lower req for reqs, and bigger
            # magnitude for invertible stats with both strictly < 0.
            if va < vb: a_better = True
            else:       b_better = True

        if a_better and b_better:
            return 0

    if a_better:
        return 1
    if b_better:
        return -1

    return 2