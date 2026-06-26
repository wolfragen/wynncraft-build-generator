"""
ingredient_filter.py

Filters raw ingredients according to Query rules.

Exact same logic as original implementation,
adapted to dense stat vectors.

Effectiveness filtering removed.
"""

from data.stats import (
    STAT_INDEX, IDX_DURABILITY, IDX_DURATION, IDX_CHARGES, FORMULA_MUL_DIV_100,
)
from data.ingredient_loader import SKILL_INDEX

import numpy as np
from numba import njit


def filter_raw_ingredients(
    ingredients_raw,
    query,
    recipe,
    cull=True,
    include_meta=False,
    recipe_lvl=None,
):
    """
    Takes all ingredients and the user Query, then returns only useful ingredients.

    `cull=True` (default) returns the keep-filtered list after the legacy
    range-aware pareto cull (back-compat for pareto_search and single-DB callers).
    `cull=False` returns the keep-filtered list WITHOUT culling — callers that
    build the per-eff-sign dual databases pass this and then call
    `split_and_cull_by_sign`, which culls each sign with a sound directional rule.

    `include_meta=True` switches to MILP-candidate mode: return EVERY legal
    ingredient — meta ingredients (non-zero posMods, positive durability) included
    — applying ONLY the legality gates (skill, and level if `recipe_lvl` is given).
    It bypasses the posMods exclusion, the positive-durability exclusion, the
    usefulness `keep` filter, and the cull, all of which are DFS-only optimisations
    that would drop ingredients an exact MILP needs for a provable optimum. The
    DFS path is unaffected: the flag defaults to False and existing callers never
    pass it. `recipe_lvl` (e.g. `RawRecipe.lvl_max`) gates over-level ingredients
    so a "proven optimal" pick is actually craftable; only consulted when
    `include_meta=True`.

    TODO(level filter): missing — ingredients with `lvl > recipe.lvl` cannot be
    used in this recipe but currently survive the filter. They get pruned at
    crafter-validation time, but here they'd be wasting cull / DFS work. Add a
    `if ing.lvl > recipe.lvl: continue` (or similar bound from RecipeRaw) once
    the recipe-level field is plumbed through. Cheap fix, modest speedup; see
    speedup_ideas.md for context.
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

    # MILP-candidate mode: legality gates only, every ingredient kept (meta too).
    if include_meta:
        out = []
        for ing in ingredients_raw:
            if skill_index is not None and not ing.skills[skill_index]:
                continue
            if recipe_lvl is not None and ing.lvl > recipe_lvl:
                continue
            out.append(ing)
        return out

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

    if not cull:
        return filtered

    n_filtered = len(filtered)
    filtered = pareto_cull_ingredients(filtered, query, recipe)
    n_culled = len(filtered)

    pct = (n_culled / n_filtered * 100) if n_filtered else 0.0
    print(f"Ingredient culling: {n_filtered} => {n_culled}, {pct:.2f}% left")
    return filtered


def pareto_cull_ingredients(ingredients, query, recipe):

    if len(ingredients) <= 1:
        return ingredients

    if query.fast_cull:
        return _pareto_cull_fast(ingredients, query, recipe)
    return _pareto_cull_exact(ingredients, query, recipe)


# ----------------------------------------------------------------------
# Per-effectiveness-sign split + sound directional cull (the "dual database")
# ----------------------------------------------------------------------
#
# WHY: the legacy `_pareto_cull_exact` `search_inv=True` branch uses
# range-containment dominance, which is UNSOUND vs the leaf's
# `0.99*max + 0.01*min` blend under inversion (a wide range [3,8] wrongly
# "dominates" a tight high range [7,8], dropping the optimal negative-eff
# filler). An ingredient's per-stat value is a V-shape in slot effectiveness:
# one ray for eff>0, one for eff<0, so two ingredients can each win on a
# different eff sign — genuinely incomparable on one shared list.
#
# FIX: split candidates into two databases by eff sign. Within a fixed sign the
# value is MONOTONE, so the plain (non-inversion) directional dual-bound
# dominance is sound. Each DB keeps only ingredients useful at that sign.
#   - positive-eff DB: cull direction = query.lower_better_proj (as-is)
#   - negative-eff DB: direction flipped (eff<0 inverts roll->contribution).
#     Durability is added raw (never eff-scaled); the cull special-cases
#     dura_proj_idx and ignores its lower_better entry, so flipping it is moot.


def split_and_cull_by_sign(ingredients, query, recipe):
    """
    Split keep-filtered ingredients into the positive-eff and negative-eff
    candidate lists (membership = "useful at that eff sign for the query") and
    cull each with the sound directional dominance for that sign.

    Returns (list_pos, list_neg) of RawIngredient. An ingredient useful at both
    signs appears in both lists and is culled independently in each.
    """
    n = len(ingredients)
    if n == 0:
        return [], []

    active = query.active_indices
    stat_count = query.stat_count

    mat_min = np.empty((n, stat_count), dtype=np.int32)
    mat_max = np.empty((n, stat_count), dtype=np.int32)
    for i, ing in enumerate(ingredients):
        mat_min[i] = ing.stats_min[active]
        mat_max[i] = ing.stats_max[active]

    useful_pos, useful_neg = _useful_masks_split(
        mat_min, mat_max,
        query.has_min_mask_proj, query.has_max_mask_proj, query.weights_proj,
        query.comp_count, query.comp_formula, query.comp_dep_offset,
        query.comp_dep_count, query.comp_dep_indices, query.comp_weight,
        query.comp_has_min, query.comp_has_max,
        query.dura_proj_idx,
    )

    list_pos = [ingredients[i] for i in range(n) if useful_pos[i]]
    list_neg = [ingredients[i] for i in range(n) if useful_neg[i]]

    lb_pos = np.ascontiguousarray(query.lower_better_proj)
    lb_neg = np.ascontiguousarray(np.logical_not(query.lower_better_proj))

    pos_culled = _cull_signed(list_pos, query, recipe, lb_pos)
    neg_culled = _cull_signed(list_neg, query, recipe, lb_neg)

    print(
        f"Dual-DB cull: filtered {n} => pos {len(list_pos)}->{len(pos_culled)}, "
        f"neg {len(list_neg)}->{len(neg_culled)}"
    )
    return pos_culled, neg_culled


def _cull_signed(ingredients, query, recipe, lower_better):
    """
    Directional pareto cull for ONE eff sign. Uses the (sound) non-inversion
    dual-bound dominance with the supplied per-stat `lower_better` direction.
    """
    if len(ingredients) <= 1:
        return ingredients

    active = query.active_indices
    stat_count = query.stat_count
    dura_proj_idx = query.dura_proj_idx
    dur_idx = IDX_DURATION if query.consumable else IDX_DURABILITY

    matrix_min = np.zeros((len(ingredients), stat_count), dtype=np.int32)
    matrix_max = np.zeros((len(ingredients), stat_count), dtype=np.int32)

    for i, ing in enumerate(ingredients):
        for j, stat_idx in enumerate(active):
            min_val = int(ing.stats_min[stat_idx])
            max_val = int(ing.stats_max[stat_idx])
            if stat_idx == dur_idx:
                min_val += recipe.scaled_dura_min
                max_val += recipe.scaled_dura_max
            matrix_min[i, j] = min_val
            matrix_max[i, j] = max_val

    # search_inv=False -> the sound directional dual-bound dominance branch.
    kept_mask = pareto_filter_ingredients(
        matrix_min, matrix_max, lower_better, dura_proj_idx, False,
    )
    return [ingredients[i] for i in range(len(ingredients)) if kept_mask[i]]


def _pareto_cull_exact(ingredients, query, recipe):
    """
    Range-aware cull. Stores BOTH bounds per (ingredient, stat), so dominance
    holds under inversion (negative slot eff flips the sign of every
    contribution and the "other" bound becomes load-bearing).
    """
    active = query.active_indices
    lower_better = query.lower_better_proj
    stat_count = query.stat_count
    dura_proj_idx = query.dura_proj_idx
    search_inv = query.search_for_inversion

    dur_idx = IDX_DURATION if query.consumable else IDX_DURABILITY

    matrix_min = np.zeros((len(ingredients), stat_count), dtype=np.int32)
    matrix_max = np.zeros((len(ingredients), stat_count), dtype=np.int32)

    for i, ing in enumerate(ingredients):

        for j, stat_idx in enumerate(active):

            min_val = int(ing.stats_min[stat_idx])
            max_val = int(ing.stats_max[stat_idx])

            # ---- apply recipe dura shift ----
            if stat_idx == dur_idx:
                min_val += recipe.scaled_dura_min
                max_val += recipe.scaled_dura_max

            matrix_min[i, j] = min_val
            matrix_max[i, j] = max_val

    kept_mask = pareto_filter_ingredients(
        matrix_min, matrix_max, lower_better, dura_proj_idx, search_inv,
    )

    return [ingredients[i] for i in range(len(ingredients)) if kept_mask[i]]


def _pareto_cull_fast(ingredients, query, recipe):
    """
    Legacy single-representative cull. Each cell stores one "best-roll" value
    along the user's preferred direction. Cheaper to compare (single int per
    pair per stat) and prunes more aggressively → smaller surviving set →
    much faster downstream search. Unsound under inversion: an ingredient
    with hpBonus=[480,550] will be treated as strictly better than [350,400]
    even though in a negative slot the inequality reverses. Use only when
    you're willing to trade correctness for speed.
    """
    active = query.active_indices
    lower_better = query.lower_better_proj
    stat_count = query.stat_count
    dura_proj_idx = query.dura_proj_idx

    dur_idx = IDX_DURATION if query.consumable else IDX_DURABILITY

    matrix = np.zeros((len(ingredients), stat_count), dtype=np.int32)

    for i, ing in enumerate(ingredients):

        for j, stat_idx in enumerate(active):

            min_val = int(ing.stats_min[stat_idx])
            max_val = int(ing.stats_max[stat_idx])

            if stat_idx == dur_idx:
                min_val += recipe.scaled_dura_min
                max_val += recipe.scaled_dura_max

            if lower_better[j]:
                if max_val > 0:
                    matrix[i, j] = min_val
                else:
                    matrix[i, j] = max_val
            elif max_val > 0:
                matrix[i, j] = max_val
            else:
                matrix[i, j] = min_val

    kept_mask = pareto_filter_ingredients_fast(matrix, lower_better, dura_proj_idx)

    return [ingredients[i] for i in range(len(ingredients)) if kept_mask[i]]


@njit(cache=True)
def pareto_filter_ingredients(matrix_min, matrix_max, lower_better, dura_proj_idx, search_inv):

    n = matrix_min.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    for i in range(n):

        if not is_kept[i]:
            continue

        for j in range(i + 1, n):

            if not is_kept[j]:
                continue

            cmp = compare_stat_vectors(
                matrix_min[i], matrix_max[i],
                matrix_min[j], matrix_max[j],
                lower_better, dura_proj_idx, search_inv,
            )

            if cmp == 1:
                is_kept[j] = False

            elif cmp == -1:
                is_kept[i] = False
                break

            elif cmp == 2:
                is_kept[j] = False

    return is_kept


@njit(cache=True)
def compare_stat_vectors(a_min, a_max, b_min, b_max, lower_better, dura_i, search_inv):
    """
    Returns:
        1  if A dominates B
       -1  if B dominates A
        2  if identical
        0  if incomparable

    Each ingredient is represented by its full roll range [min, max] per stat,
    so dominance can be checked correctly even when slot effectiveness flips
    sign (search_inv=True).

    With inversion:
        Optimal contribution at slot eff e is e·max for e>0 and e·min for e<0
        (the search picks the best roll per stat). For A to dominate B at
        EVERY possible slot eff we need a_max ≥ b_max (positive-eff side) AND
        a_min ≤ b_min (negative-eff side), i.e., A's range contains B's.

    Without inversion:
        Only positive eff is possible, so only the user-preferred bound
        matters. For higher-better stats, both bounds must be ≥ (so B can't
        roll a higher value than A's best); for lower-better, both ≤.

    `dura_i` is the projected column of durability/duration (or -1). Dura is
    treated as strictly higher-is-better on the max bound; the inversion
    branch isn't applied there (no meaningful negative-eff dura semantics in
    the current search).
    """

    a_better = False
    b_better = False

    for i in range(len(a_min)):

        a_mn = a_min[i]
        a_mx = a_max[i]
        b_mn = b_min[i]
        b_mx = b_max[i]

        if a_mn == b_mn and a_mx == b_mx:
            continue

        # Dura/duration: strictly higher = better. Compare on max, then min.
        if i == dura_i:
            if a_mx > b_mx:
                a_better = True
            elif a_mx < b_mx:
                b_better = True
            elif a_mn > b_mn:
                a_better = True
            else:
                b_better = True
            if a_better and b_better:
                return 0
            continue

        if search_inv:
            # Range containment: A dominates iff A's range contains B's.
            # Identical ranges were already skipped above.
            if a_mn <= b_mn and a_mx >= b_mx:
                a_better = True
            elif b_mn <= a_mn and b_mx >= a_mx:
                b_better = True
            else:
                return 0
        else:
            lb = lower_better[i]
            if lb:
                # Lower-better: A dom iff both bounds ≤ B's.
                if a_mn <= b_mn and a_mx <= b_mx:
                    a_better = True
                elif b_mn <= a_mn and b_mx <= a_mx:
                    b_better = True
                else:
                    return 0
            else:
                # Higher-better: A dom iff both bounds ≥ B's.
                if a_mn >= b_mn and a_mx >= b_mx:
                    a_better = True
                elif b_mn >= a_mn and b_mx >= a_mx:
                    b_better = True
                else:
                    return 0

        if a_better and b_better:
            return 0

    if a_better:
        return 1
    if b_better:
        return -1

    return 2


# ----------------------------------------------------------------------
# Legacy fast cull — single-representative, sign-mismatch guarded.
# Used when query.fast_cull=True. Trades soundness under inversion for
# speed (smaller surviving ingredient set, fewer pairwise comparisons).
# ----------------------------------------------------------------------

@njit(cache=True)
def pareto_filter_ingredients_fast(matrix, lower_better, dura_proj_idx):

    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    for i in range(n):

        if not is_kept[i]:
            continue

        for j in range(i + 1, n):

            if not is_kept[j]:
                continue

            cmp = compare_stat_vectors_fast(
                matrix[i], matrix[j], lower_better, dura_proj_idx,
            )

            if cmp == 1:
                is_kept[j] = False

            elif cmp == -1:
                is_kept[i] = False
                break

            elif cmp == 2:
                is_kept[j] = False

    return is_kept


@njit(cache=True)
def compare_stat_vectors_fast(a, b, lower_better, dura_i):
    """
    Returns:
        1  if A dominates B
       -1  if B dominates A
        2  if identical
        0  if incomparable
    """

    a_better = False
    b_better = False

    for i in range(len(a)):

        va = a[i]
        vb = b[i]

        if va == vb:
            continue

        if i == dura_i:
            if va > vb: a_better = True
            else:       b_better = True
            if a_better and b_better:
                return 0
            continue

        if (va > 0 and vb < 0) or (va < 0 and vb > 0):
            return 0

        lb = lower_better[i]

        if va > 0 or vb > 0:
            if (va > vb) != lb:
                a_better = True
            else:
                b_better = True
        else:
            if not lb and (va == 0 or vb == 0):
                return 0
            if va < vb: a_better = True
            else:       b_better = True

        if a_better and b_better:
            return 0

    if a_better:
        return 1
    if b_better:
        return -1

    return 2


# ----------------------------------------------------------------------
# Membership helper for the dual-database split.
# ----------------------------------------------------------------------

@njit(cache=True)
def _useful_masks_split(
    db_stat_min, db_stat_max,
    has_min_mask, has_max_mask, weights,
    comp_count, comp_formula, comp_dep_offset, comp_dep_count,
    comp_dep_indices, comp_weight, comp_has_min, comp_has_max,
    dura_idx,
):
    """
    Per-(ingredient, eff sign) usefulness, over PROJECTED stats. Mirrors
    `core.search_engine._compute_useful_masks` but skips the durability/duration
    column (`dura_idx`) so a stat that is never effectiveness-scaled cannot push
    an ingredient into a sign-database where it has no eff-scaled use.

    `useful_pos[i]` True iff ingredient i can help some active stat/composite at
    eff > 0; `useful_neg[i]` symmetric for eff < 0.
    """
    N = db_stat_min.shape[0]
    S = db_stat_min.shape[1]

    s_higher = np.zeros(S, dtype=np.bool_)
    s_lower = np.zeros(S, dtype=np.bool_)
    for s in range(S):
        if has_min_mask[s] or weights[s] > 0.0:
            s_higher[s] = True
        if has_max_mask[s] or weights[s] < 0.0:
            s_lower[s] = True

    for c in range(comp_count):
        cw = comp_weight[c]
        ch_min = comp_has_min[c]
        ch_max = comp_has_max[c]
        if cw == 0.0 and (not ch_min) and (not ch_max):
            continue
        f = comp_formula[c]
        nonmonotone = (f == FORMULA_MUL_DIV_100)
        off = comp_dep_offset[c]
        cnt = comp_dep_count[c]
        for k in range(cnt):
            dep_s = comp_dep_indices[off + k]
            if nonmonotone:
                s_higher[dep_s] = True
                s_lower[dep_s] = True
            else:
                if cw > 0.0 or ch_min:
                    s_higher[dep_s] = True
                if cw < 0.0 or ch_max:
                    s_lower[dep_s] = True

    useful_pos = np.zeros(N, dtype=np.bool_)
    useful_neg = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        for s in range(S):
            if s == dura_idx:
                continue
            can_pos = db_stat_max[i, s] > 0
            can_neg = db_stat_min[i, s] < 0
            if can_pos:
                if s_higher[s]:
                    useful_pos[i] = True
                if s_lower[s]:
                    useful_neg[i] = True
            if can_neg:
                if s_lower[s]:
                    useful_pos[i] = True
                if s_higher[s]:
                    useful_neg[i] = True
            if useful_pos[i] and useful_neg[i]:
                break
    return useful_pos, useful_neg