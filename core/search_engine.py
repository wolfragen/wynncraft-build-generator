"""
search_engine.py
"""

import numpy as np
from numba import njit, prange
from numba import types
from queue import Queue
from threading import Thread
from time import time

from data.stats import FORMULA_MUL_DIV_100, FORMULA_RAW_TO_PCT, FORMULA_EHP, FORMULA_EHPR
from data.skillpoint_lookup import SKP_HEADLINE_PCT, SKP_DEF, SKP_AGI, SKP_MAX

# Numba captures module-level numpy arrays as constants (read-only refs); the
# table is precomputed once at import time and never mutated, so this is safe
# even with cache=True.
_SKP_DEF_ROW = SKP_HEADLINE_PCT[SKP_DEF].copy()  # shape (151,) float64
_SKP_AGI_ROW = SKP_HEADLINE_PCT[SKP_AGI].copy()


# Explicit signature so the recursive dfs can be persisted to disk with
# cache=True — numba cannot otherwise resolve the self-reference symbol.
_DFS_SIG = types.void(
    types.int64,              # depth
    types.int64,              # start_index
    types.int64,              # k
    types.int32[::1],         # ingredients
    types.int32[::1],         # current_min
    types.int32[::1],         # current_max
    types.float64[::1],       # best_score_ref
    types.int32[::1],         # best_solution
    types.int16[:, ::1],      # db_stat_min
    types.int16[:, ::1],      # db_stat_max
    types.Array(types.bool_, 2, "C"),  # db_contrib_pos_mask
    types.Array(types.bool_, 2, "C"),  # db_contrib_neg_mask
    types.int64,              # db_count
    types.int32[::1],         # meta_void_eff
    types.int64,              # dura_idx
    types.Array(types.bool_, 1, "C"),  # has_min_mask
    types.Array(types.bool_, 1, "C"),  # has_max_mask
    types.Array(types.bool_, 1, "C"),  # pos_weight_mask
    types.Array(types.bool_, 1, "C"),  # neg_weight_mask
    types.int32[::1],         # min_vals
    types.int32[::1],         # max_vals
    types.float32[::1],       # weights
    types.int64[::1],         # total_searched
    types.int32[:, ::1],      # future_max_ub
    types.int32[:, ::1],      # future_min_ub
    types.int32[:, ::1],      # future_max_lb
    types.int32[:, ::1],      # future_min_lb
    types.int64,              # comp_count
    types.int32[::1],         # comp_formula
    types.int32[::1],         # comp_dep_a_proj
    types.int32[::1],         # comp_dep_b_proj
    types.int32[::1],         # comp_dep_c_proj
    types.int32[::1],         # comp_dep_d_proj
    types.int32[::1],         # comp_min
    types.int32[::1],         # comp_max
    types.Array(types.bool_, 1, "C"),  # comp_has_min
    types.Array(types.bool_, 1, "C"),  # comp_has_max
    types.float32[::1],       # comp_weight
)


# ============================================================
# Composite-stat 4-corner bound on a*b//100 over rectangle [a_lo,a_hi]x[b_lo,b_hi].
# ============================================================

@njit(cache=True)
def _product_bounds_div100(a_lo, a_hi, b_lo, b_hi):
    """
    Bounds on ((a*b) // 100) where a ∈ [a_lo, a_hi], b ∈ [b_lo, b_hi].
    The product of two real-valued intervals has its extrema at the rectangle
    corners, regardless of sign — so evaluate 4 corners and take min/max.
    Intermediate math in int64 to avoid int32 overflow on wide builds.
    """
    al = np.int64(a_lo)
    ah = np.int64(a_hi)
    bl = np.int64(b_lo)
    bh = np.int64(b_hi)
    p1 = al * bl
    p2 = al * bh
    p3 = ah * bl
    p4 = ah * bh
    lo = p1
    if p2 < lo: lo = p2
    if p3 < lo: lo = p3
    if p4 < lo: lo = p4
    hi = p1
    if p2 > hi: hi = p2
    if p3 > hi: hi = p3
    if p4 > hi: hi = p4
    return lo // 100, hi // 100


# ============================================================
# rawToPct: wynnbuilder's sign-asymmetric raw/pct combiner.
#   raw > 0:  (raw * (100 + delta)) // 100
#   raw < 0:  min(0, (raw * (100 - delta)) // 100)
#   raw = 0:  0
# `delta` is stored as a signed delta (not 100+delta).
# ============================================================

@njit(cache=True)
def _raw_to_pct(raw, delta):
    r = np.int64(raw)
    d = np.int64(delta)
    if r > 0:
        return (r * (100 + d)) // 100
    if r < 0:
        v = (r * (100 - d)) // 100
        if v < 0:
            return v
        return np.int64(0)
    return np.int64(0)


@njit(cache=True)
def _raw_to_pct_bounds(raw_lo, raw_hi, delta_lo, delta_hi):
    """
    Admissible bounds on _raw_to_pct(raw, delta) over the rectangle
    [raw_lo, raw_hi] × [delta_lo, delta_hi]. Split on sign of raw:
      - positive half (raw ≥ 0): bilinear raw*(100+delta), floor-div at end.
      - negative half (raw ≤ 0): bilinear g = raw*(100-delta), clamped by min(0, g).
    Floor(max(·)) == max(floor(·)) for monotone floor, so the bound is consistent
    with the leaf's // 100 floor (not with wynnbuilder's float math — ±1 divergence
    on negative intermediate values, same divergence as the leaf, so consistent).
    """
    dl = np.int64(delta_lo)
    dh = np.int64(delta_hi)

    first = True
    lo = np.int64(0)
    hi = np.int64(0)

    if raw_hi > 0:
        rl = np.int64(raw_lo)
        if rl < 0:
            rl = np.int64(0)
        rh = np.int64(raw_hi)
        c1 = rl * (100 + dl)
        c2 = rl * (100 + dh)
        c3 = rh * (100 + dl)
        c4 = rh * (100 + dh)
        p_lo = c1
        if c2 < p_lo: p_lo = c2
        if c3 < p_lo: p_lo = c3
        if c4 < p_lo: p_lo = c4
        p_hi = c1
        if c2 > p_hi: p_hi = c2
        if c3 > p_hi: p_hi = c3
        if c4 > p_hi: p_hi = c4
        p_lo = p_lo // 100
        p_hi = p_hi // 100
        lo = p_lo
        hi = p_hi
        first = False

    if raw_lo < 0:
        rl = np.int64(raw_lo)
        rh = np.int64(raw_hi)
        if rh > 0:
            rh = np.int64(0)
        c1 = rl * (100 - dl)
        c2 = rl * (100 - dh)
        c3 = rh * (100 - dl)
        c4 = rh * (100 - dh)
        g_lo = c1
        if c2 < g_lo: g_lo = c2
        if c3 < g_lo: g_lo = c3
        if c4 < g_lo: g_lo = c4
        g_hi = c1
        if c2 > g_hi: g_hi = c2
        if c3 > g_hi: g_hi = c3
        if c4 > g_hi: g_hi = c4
        g_lo = g_lo // 100
        g_hi = g_hi // 100
        # Clamp with min(0, ·)
        if g_lo > 0: g_lo = np.int64(0)
        if g_hi > 0: g_hi = np.int64(0)
        if first:
            lo = g_lo
            hi = g_hi
            first = False
        else:
            if g_lo < lo: lo = g_lo
            if g_hi > hi: hi = g_hi

    return lo, hi


# ============================================================
# ehp / ehpr (build-level effective HP and HP regen)
# Simplified from wynnbuilder's getDefenseStats — assumes:
#   - level-base hp = 0 (only hpBonus contributes to totalHp)
#   - classDef = 0 (no weapon-class adjustment)
#   - defMult = 1 (no potion/buff multipliers)
#   - agiDef = 0 (so agi_reduction = 1.0)
# Resulting formulas:
#   ehp  = max(5, hpBonus) / (1 - (1 - agi_pct) * def_pct)
#   ehpr = rawToPct(hprRaw, hprPct) / (1 - (1 - agi_pct) * def_pct)
# def_pct, agi_pct from SKP_HEADLINE_PCT (clamp counts to [0, 150]).
# ============================================================


@njit(cache=True)
def _clamp_skp(c):
    if c < 0:
        return np.int64(0)
    if c > SKP_MAX:
        return np.int64(SKP_MAX)
    return np.int64(c)


@njit(cache=True)
def _denom_range(def_lo, def_hi, agi_lo, agi_hi):
    """
    Range of D = 1 - (1 - agi_pct) * def_pct over (def_count, agi_count) ranges.
    def_pct, agi_pct nondecreasing in skillpoint count, both in [0, ~0.81].
    Bilinear: D_min when (1-a)*d max → (a_min, d_max); D_max at (a_max, d_min).
    D > 0 always in this domain (max product ≤ 1 * ~0.7 = 0.7).
    """
    dl = _clamp_skp(def_lo)
    dh = _clamp_skp(def_hi)
    al = _clamp_skp(agi_lo)
    ah = _clamp_skp(agi_hi)
    def_pct_lo = _SKP_DEF_ROW[dl]
    def_pct_hi = _SKP_DEF_ROW[dh]
    agi_pct_lo = _SKP_AGI_ROW[al]
    agi_pct_hi = _SKP_AGI_ROW[ah]
    d_min = 1.0 - (1.0 - agi_pct_lo) * def_pct_hi
    d_max = 1.0 - (1.0 - agi_pct_hi) * def_pct_lo
    return d_min, d_max


@njit(cache=True)
def _ehp_bounds(hp_lo, hp_hi, def_lo, def_hi, agi_lo, agi_hi):
    """
    Bounds on max(5, hpBonus) / (1 - (1-agi_pct)*def_pct) over the 3D rectangle.
    Used at both leaf (cmin/cmax of the build under independent rolls) and
    UB sites (with future_*_lb/ub-expanded ranges).
    """
    if hp_lo < 5: hp_lo = 5
    if hp_hi < 5: hp_hi = 5
    d_min, d_max = _denom_range(def_lo, def_hi, agi_lo, agi_hi)
    # Positive totalHp, positive D: max ehp = hp_hi / d_min, min = hp_lo / d_max.
    cmax_f = hp_hi / d_min
    cmin_f = hp_lo / d_max
    return np.int64(cmin_f), np.int64(cmax_f)


@njit(cache=True)
def _ehpr_bounds(raw_lo, raw_hi, pct_lo, pct_hi, def_lo, def_hi, agi_lo, agi_hi):
    """
    Bounds on rawToPct(hprRaw, hprPct) / (1 - (1-agi_pct)*def_pct).
    totalHpr range comes from _raw_to_pct_bounds (sign-aware, can be negative).
    Then 4-corner bound on (totalHpr, D) — sign of totalHpr decides which D
    extreme gives max/min of the quotient.
    """
    hpr_lo, hpr_hi = _raw_to_pct_bounds(raw_lo, raw_hi, pct_lo, pct_hi)
    d_min, d_max = _denom_range(def_lo, def_hi, agi_lo, agi_hi)
    c1 = hpr_lo / d_min
    c2 = hpr_lo / d_max
    c3 = hpr_hi / d_min
    c4 = hpr_hi / d_max
    cmin_f = c1
    if c2 < cmin_f: cmin_f = c2
    if c3 < cmin_f: cmin_f = c3
    if c4 < cmin_f: cmin_f = c4
    cmax_f = c1
    if c2 > cmax_f: cmax_f = c2
    if c3 > cmax_f: cmax_f = c3
    if c4 > cmax_f: cmax_f = c4
    return np.int64(cmin_f), np.int64(cmax_f)


# ============================================================
# Precompute per-meta-set bounds for B&B pruning
# ============================================================

@njit(cache=True)
def _precompute_bounds(db_stat_min, db_stat_max, meta_void_eff, void_count, dura_idx):
    """
    Build suffix-sum arrays capturing the best/worst additional contribution
    the remaining slots can add to current_max[s] and current_min[s]. Semantics
    match the running sums used in dfs: current_max uses db_stat_max across
    ingredients, current_min uses db_stat_min.

    Returns four (void_count+1, S) int32 arrays:
      future_max_ub[d, s] — max extra contribution to current_max from slots d..k-1
      future_min_ub[d, s] — max extra contribution to current_min from slots d..k-1
      future_max_lb[d, s] — min extra contribution to current_max from slots d..k-1
      future_min_lb[d, s] — min extra contribution to current_min from slots d..k-1
    """
    S = db_stat_min.shape[1]
    N = db_stat_min.shape[0]
    k = void_count

    future_max_ub = np.zeros((k + 1, S), dtype=np.int32)
    future_min_ub = np.zeros((k + 1, S), dtype=np.int32)
    future_max_lb = np.zeros((k + 1, S), dtype=np.int32)
    future_min_lb = np.zeros((k + 1, S), dtype=np.int32)

    # Walk slots from last to first; accumulate the per-slot best/worst
    # single-ingredient contribution into the suffix sum.
    for d in range(k - 1, -1, -1):
        eff = meta_void_eff[d]
        for s in range(S):
            if s == dura_idx:
                # Dura: no eff multiplier, contribution is just db_stat_X[i, dura].
                slot_best_max = db_stat_max[0, s]
                slot_worst_max = db_stat_max[0, s]
                slot_best_min = db_stat_min[0, s]
                slot_worst_min = db_stat_min[0, s]
                for i in range(1, N):
                    v = db_stat_max[i, s]
                    if v > slot_best_max:
                        slot_best_max = v
                    if v < slot_worst_max:
                        slot_worst_max = v
                    v = db_stat_min[i, s]
                    if v > slot_best_min:
                        slot_best_min = v
                    if v < slot_worst_min:
                        slot_worst_min = v
            else:
                # Stat with effectiveness multiplier. The sign of eff flips
                # which ingredient side wins.
                v0 = (db_stat_max[0, s] * eff) // 100
                slot_best_max = v0
                slot_worst_max = v0
                v0 = (db_stat_min[0, s] * eff) // 100
                slot_best_min = v0
                slot_worst_min = v0
                for i in range(1, N):
                    v = (db_stat_max[i, s] * eff) // 100
                    if v > slot_best_max:
                        slot_best_max = v
                    if v < slot_worst_max:
                        slot_worst_max = v
                    v = (db_stat_min[i, s] * eff) // 100
                    if v > slot_best_min:
                        slot_best_min = v
                    if v < slot_worst_min:
                        slot_worst_min = v

            future_max_ub[d, s] = future_max_ub[d + 1, s] + slot_best_max
            future_min_ub[d, s] = future_min_ub[d + 1, s] + slot_best_min
            future_max_lb[d, s] = future_max_lb[d + 1, s] + slot_worst_max
            future_min_lb[d, s] = future_min_lb[d + 1, s] + slot_worst_min

    return future_max_ub, future_min_ub, future_max_lb, future_min_lb


# ============================================================
# DFS (numba)
# ============================================================

@njit(_DFS_SIG, cache=True)
def dfs(
    depth,
    start_index,
    k,
    ingredients,
    current_min,
    current_max,
    best_score_ref,
    best_solution,
    db_stat_min,
    db_stat_max,
    db_contrib_pos_mask,
    db_contrib_neg_mask,
    db_count,
    meta_void_eff,
    dura_idx,
    has_min_mask,
    has_max_mask,
    pos_weight_mask,
    neg_weight_mask,
    min_vals,
    max_vals,
    weights,
    total_searched,
    future_max_ub,
    future_min_ub,
    future_max_lb,
    future_min_lb,
    comp_count,
    comp_formula,
    comp_dep_a_proj,
    comp_dep_b_proj,
    comp_dep_c_proj,
    comp_dep_d_proj,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
):
    # Leaf
    if depth == k:
        total_searched[0] += 1

        score = 0.0

        # Evaluate directly using current stats
        for s in range(len(current_min)):

            min_v = current_min[s]
            max_v = current_max[s]

            if has_min_mask[s] and max_v < min_vals[s]:
                return

            if has_max_mask[s] and min_v > max_vals[s]:
                return

            score += weights[s] * max_v * 0.99
            score += weights[s] * min_v * 0.01

        # Composite stats: compute on the finalized build, check constraints,
        # add weighted contribution.
        for c in range(comp_count):
            a = comp_dep_a_proj[c]
            b = comp_dep_b_proj[c]
            f = comp_formula[c]
            if f == FORMULA_MUL_DIV_100:
                cmax = (np.int64(current_max[a]) * np.int64(current_max[b])) // 100
                cmin = (np.int64(current_min[a]) * np.int64(current_min[b])) // 100
            elif f == FORMULA_RAW_TO_PCT:
                cmax = _raw_to_pct(current_max[a], current_max[b])
                cmin = _raw_to_pct(current_min[a], current_min[b])
            elif f == FORMULA_EHP:
                cc = comp_dep_c_proj[c]
                cmin, cmax = _ehp_bounds(
                    current_min[a], current_max[a],
                    current_min[b], current_max[b],
                    current_min[cc], current_max[cc],
                )
            else:  # FORMULA_EHPR
                cc = comp_dep_c_proj[c]
                dd = comp_dep_d_proj[c]
                cmin, cmax = _ehpr_bounds(
                    current_min[a], current_max[a],
                    current_min[b], current_max[b],
                    current_min[cc], current_max[cc],
                    current_min[dd], current_max[dd],
                )

            if comp_has_min[c] and cmax < comp_min[c]:
                return
            if comp_has_max[c] and cmin > comp_max[c]:
                return

            score += comp_weight[c] * (cmax * 0.99 + cmin * 0.01)

        if score > best_score_ref[0]:
            best_score_ref[0] = score
            for i in range(k):
                best_solution[i] = ingredients[i]

        return


    for i in range(start_index, db_count):

        eff = meta_void_eff[depth]
        
        # ============================================================
        # PRUNING
        # ============================================================
        
        useful = False
        eff_is_positive = eff > 0
        for s in range(len(current_min)):

            # contrib_pos (ing stat can be > 0):
            #   eff > 0 → product > 0 → relevant to has_min / positive weight
            #   eff < 0 → product < 0 → relevant to has_max / negative weight
            if db_contrib_pos_mask[i, s]:
                if eff_is_positive and (has_min_mask[s] or pos_weight_mask[s]):
                    useful = True
                    break
                if (not eff_is_positive) and (has_max_mask[s] or neg_weight_mask[s]):
                    useful = True
                    break

            # contrib_neg (ing stat can be < 0):
            #   eff > 0 → product < 0 → relevant to has_max / negative weight
            #   eff < 0 → product > 0 → relevant to has_min / positive weight
            if db_contrib_neg_mask[i, s]:
                if eff_is_positive and (has_max_mask[s] or neg_weight_mask[s]):
                    useful = True
                    break
                if (not eff_is_positive) and (has_min_mask[s] or pos_weight_mask[s]):
                    useful = True
                    break

        if not useful:
            continue
        
        # ---- dura pruning ----
        if current_max[dura_idx] + db_stat_min[i, dura_idx] < min_vals[dura_idx]:
            continue
        
        # ============================================================
        # 
        # ============================================================

        ingredients[depth] = i

        # Apply ingredient contribution
        for s in range(len(current_min)):
            if(s == dura_idx):
                current_min[s] += db_stat_min[i, s]
                current_max[s] += db_stat_max[i, s]
            else:
                current_min[s] += (db_stat_min[i, s] * eff) //100
                current_max[s] += (db_stat_max[i, s] * eff) //100

        # =========================
        # SCORE UB PRUNING (branch-and-bound)
        # =========================
        # Upper-bound the final score assuming remaining slots pick the best
        # ingredient per-stat (optimistic, stats treated independently). For
        # weights>0 we use UB on current_max/min; for weights<0 we use LB
        # (since negative weight * larger value = smaller score).
        ub_score = 0.0
        S = len(current_min)
        next_depth = depth + 1
        for s in range(S):
            w = weights[s]
            if w > 0.0:
                max_v = current_max[s] + future_max_ub[next_depth, s]
                min_v = current_min[s] + future_min_ub[next_depth, s]
                ub_score += w * (max_v * 0.99 + min_v * 0.01)
            elif w < 0.0:
                max_v = current_max[s] + future_max_lb[next_depth, s]
                min_v = current_min[s] + future_min_lb[next_depth, s]
                ub_score += w * (max_v * 0.99 + min_v * 0.01)

        # Composite UB: per-formula admissible bound on the final comp value,
        # using the (current_*[s] + future_*_*[next_depth, s]) range for each dep.
        # Loose vs true joint feasible set, but correct.
        for c in range(comp_count):
            a = comp_dep_a_proj[c]
            b = comp_dep_b_proj[c]
            w = comp_weight[c]
            if w == 0.0:
                continue
            f = comp_formula[c]

            a_max_lo = current_max[a] + future_max_lb[next_depth, a]
            a_max_hi = current_max[a] + future_max_ub[next_depth, a]
            b_max_lo = current_max[b] + future_max_lb[next_depth, b]
            b_max_hi = current_max[b] + future_max_ub[next_depth, b]
            a_min_lo = current_min[a] + future_min_lb[next_depth, a]
            a_min_hi = current_min[a] + future_min_ub[next_depth, a]
            b_min_lo = current_min[b] + future_min_lb[next_depth, b]
            b_min_hi = current_min[b] + future_min_ub[next_depth, b]

            if f == FORMULA_MUL_DIV_100:
                pmax_lo, pmax_hi = _product_bounds_div100(a_max_lo, a_max_hi, b_max_lo, b_max_hi)
                pmin_lo, pmin_hi = _product_bounds_div100(a_min_lo, a_min_hi, b_min_lo, b_min_hi)
            elif f == FORMULA_RAW_TO_PCT:
                pmax_lo, pmax_hi = _raw_to_pct_bounds(a_max_lo, a_max_hi, b_max_lo, b_max_hi)
                pmin_lo, pmin_hi = _raw_to_pct_bounds(a_min_lo, a_min_hi, b_min_lo, b_min_hi)
            elif f == FORMULA_EHP:
                cc = comp_dep_c_proj[c]
                c_max_lo = current_max[cc] + future_max_lb[next_depth, cc]
                c_max_hi = current_max[cc] + future_max_ub[next_depth, cc]
                c_min_lo = current_min[cc] + future_min_lb[next_depth, cc]
                c_min_hi = current_min[cc] + future_min_ub[next_depth, cc]
                pmax_lo, pmax_hi = _ehp_bounds(a_max_lo, a_max_hi, b_max_lo, b_max_hi, c_max_lo, c_max_hi)
                pmin_lo, pmin_hi = _ehp_bounds(a_min_lo, a_min_hi, b_min_lo, b_min_hi, c_min_lo, c_min_hi)
            else:  # FORMULA_EHPR
                cc = comp_dep_c_proj[c]
                dd = comp_dep_d_proj[c]
                c_max_lo = current_max[cc] + future_max_lb[next_depth, cc]
                c_max_hi = current_max[cc] + future_max_ub[next_depth, cc]
                d_max_lo = current_max[dd] + future_max_lb[next_depth, dd]
                d_max_hi = current_max[dd] + future_max_ub[next_depth, dd]
                c_min_lo = current_min[cc] + future_min_lb[next_depth, cc]
                c_min_hi = current_min[cc] + future_min_ub[next_depth, cc]
                d_min_lo = current_min[dd] + future_min_lb[next_depth, dd]
                d_min_hi = current_min[dd] + future_min_ub[next_depth, dd]
                pmax_lo, pmax_hi = _ehpr_bounds(
                    a_max_lo, a_max_hi, b_max_lo, b_max_hi,
                    c_max_lo, c_max_hi, d_max_lo, d_max_hi,
                )
                pmin_lo, pmin_hi = _ehpr_bounds(
                    a_min_lo, a_min_hi, b_min_lo, b_min_hi,
                    c_min_lo, c_min_hi, d_min_lo, d_min_hi,
                )

            if w > 0.0:
                ub_score += w * (pmax_hi * 0.99 + pmin_hi * 0.01)
            else:
                ub_score += w * (pmax_lo * 0.99 + pmin_lo * 0.01)

        if ub_score <= best_score_ref[0]:
            # Undo and skip — no descendant can beat the current best.
            for s in range(len(current_min)):
                if s == dura_idx:
                    current_min[s] -= db_stat_min[i, s]
                    current_max[s] -= db_stat_max[i, s]
                else:
                    current_min[s] -= (db_stat_min[i, s] * eff) // 100
                    current_max[s] -= (db_stat_max[i, s] * eff) // 100
            continue

        # Permutation kill: only re-use the same start index when the next
        # void slot has the same effectiveness (slots are sorted desc by eff),
        # otherwise the slot is distinguishable and any ordering is allowed.
        # When depth+1 == k there's no next slot — the recursion will hit the
        # leaf and `start_index` is unused, so the value doesn't matter.
        next_start = 0
        if k == 6 or (depth + 1 < k and eff == meta_void_eff[depth + 1]):
            next_start = i

        dfs(
            depth + 1,
            next_start,
            k,
            ingredients,
            current_min,
            current_max,
            best_score_ref,
            best_solution,
            db_stat_min,
            db_stat_max,
            db_contrib_pos_mask,
            db_contrib_neg_mask,
            db_count,
            meta_void_eff,
            dura_idx,
            has_min_mask,
            has_max_mask,
            pos_weight_mask,
            neg_weight_mask,
            min_vals,
            max_vals,
            weights,
            total_searched,
            future_max_ub,
            future_min_ub,
            future_max_lb,
            future_min_lb,
            comp_count,
            comp_formula,
            comp_dep_a_proj,
            comp_dep_b_proj,
            comp_dep_c_proj,
            comp_dep_d_proj,
            comp_min,
            comp_max,
            comp_has_min,
            comp_has_max,
            comp_weight,
        )

        # Undo
        for s in range(len(current_min)):
            if(s == dura_idx):
                current_min[s] -= db_stat_min[i, s]
                current_max[s] -= db_stat_max[i, s]
            else:
                current_min[s] -= (db_stat_min[i, s] * eff) //100
                current_max[s] -= (db_stat_max[i, s] * eff) //100


# ============================================================
# Specialized fast path: k=1 (one void slot, no recursion needed)
# ============================================================
# With a single void slot the DFS is just "pick the best ingredient for this
# set". Recursion + per-set allocations + per-set precompute_bounds add
# substantial overhead over 365K sets. This specialization collapses the inner
# work to a dense double loop with zero heap allocation per set.

@njit(parallel=True, cache=True)
def _search_meta_batch_k1(
    void_eff_matrix,
    base_min_matrix,
    base_max_matrix,
    db_stat_min,
    db_stat_max,
    db_count,
    dura_idx,
    has_min_mask,
    has_max_mask,
    min_vals,
    max_vals,
    weights,
    init_best_score,
    comp_count,
    comp_formula,
    comp_dep_a_proj,
    comp_dep_b_proj,
    comp_dep_c_proj,
    comp_dep_d_proj,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
):
    M = base_min_matrix.shape[0]
    S = base_min_matrix.shape[1]

    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, 1), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    for m in prange(M):
        eff = void_eff_matrix[m, 0]
        local_best = init_best_score
        local_best_i = -1
        local_searched = 0

        # Per-iter storage so composite can reread dep values after the base loop.
        final_min = np.empty(S, dtype=np.int32)
        final_max = np.empty(S, dtype=np.int32)

        for i in range(db_count):
            feasible = True
            score = 0.0
            for s in range(S):
                if s == dura_idx:
                    min_v = base_min_matrix[m, s] + db_stat_min[i, s]
                    max_v = base_max_matrix[m, s] + db_stat_max[i, s]
                else:
                    min_v = base_min_matrix[m, s] + (db_stat_min[i, s] * eff) // 100
                    max_v = base_max_matrix[m, s] + (db_stat_max[i, s] * eff) // 100

                final_min[s] = min_v
                final_max[s] = max_v

                if has_min_mask[s] and max_v < min_vals[s]:
                    feasible = False
                    break
                if has_max_mask[s] and min_v > max_vals[s]:
                    feasible = False
                    break

                score += weights[s] * (max_v * 0.99 + min_v * 0.01)

            if not feasible:
                continue

            # Composite constraint + score
            for c in range(comp_count):
                a = comp_dep_a_proj[c]
                b = comp_dep_b_proj[c]
                f = comp_formula[c]
                if f == FORMULA_MUL_DIV_100:
                    cmax = (np.int64(final_max[a]) * np.int64(final_max[b])) // 100
                    cmin = (np.int64(final_min[a]) * np.int64(final_min[b])) // 100
                elif f == FORMULA_RAW_TO_PCT:
                    cmax = _raw_to_pct(final_max[a], final_max[b])
                    cmin = _raw_to_pct(final_min[a], final_min[b])
                elif f == FORMULA_EHP:
                    cc = comp_dep_c_proj[c]
                    cmin, cmax = _ehp_bounds(
                        final_min[a], final_max[a],
                        final_min[b], final_max[b],
                        final_min[cc], final_max[cc],
                    )
                else:  # FORMULA_EHPR
                    cc = comp_dep_c_proj[c]
                    dd = comp_dep_d_proj[c]
                    cmin, cmax = _ehpr_bounds(
                        final_min[a], final_max[a],
                        final_min[b], final_max[b],
                        final_min[cc], final_max[cc],
                        final_min[dd], final_max[dd],
                    )
                if comp_has_min[c] and cmax < comp_min[c]:
                    feasible = False
                    break
                if comp_has_max[c] and cmin > comp_max[c]:
                    feasible = False
                    break
                score += comp_weight[c] * (cmax * 0.99 + cmin * 0.01)

            if not feasible:
                continue

            local_searched += 1
            if score > local_best:
                local_best = score
                local_best_i = i

        best_scores[m] = local_best
        if local_best_i >= 0:
            best_solutions[m, 0] = local_best_i
        per_m_searched[m] = local_searched

    # Reduce: global best + total searched.
    best_score = -1e18
    best_meta_index = -1
    best_solution_global = np.zeros(1, dtype=np.int32)
    for m in range(M):
        if best_scores[m] > best_score:
            best_score = best_scores[m]
            best_meta_index = m
            best_solution_global[0] = best_solutions[m, 0]

    total = 0
    for m in range(M):
        total += per_m_searched[m]

    return best_score, best_meta_index, best_solution_global, total


# ============================================================
# Specialized fast path: k=2 (two void slots, one-depth unrolled)
# ============================================================
# With two void slots we keep B&B alive (upper bound at depth=0 uses the best
# possible contribution from slot 1 to cut whole i0 branches). Both depths are
# inlined: zero recursion, zero per-set numpy allocation.

@njit(parallel=True, cache=True)
def _search_meta_batch_k2(
    void_eff_matrix,
    base_min_matrix,
    base_max_matrix,
    db_stat_min,
    db_stat_max,
    db_count,
    dura_idx,
    has_min_mask,
    has_max_mask,
    min_vals,
    max_vals,
    weights,
    init_best_score,
    comp_count,
    comp_formula,
    comp_dep_a_proj,
    comp_dep_b_proj,
    comp_dep_c_proj,
    comp_dep_d_proj,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
):
    M = base_min_matrix.shape[0]
    S = base_min_matrix.shape[1]

    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, 2), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    for m in prange(M):
        eff0 = void_eff_matrix[m, 0]
        eff1 = void_eff_matrix[m, 1]
        same_eff = (eff0 == eff1)

        # ---- Precompute slot-1 per-stat bounds (one-time O(N*S) per m).
        # slot1_best_max[s] = max over i of db_stat_max[i, s] * eff1 // 100 (direct for dura).
        # slot1_best_min[s] = max over i of db_stat_min[i, s] * eff1 // 100.
        # Used to upper-bound the UB of the final score after picking i0.
        slot1_best_max = np.empty(S, dtype=np.int32)
        slot1_best_min = np.empty(S, dtype=np.int32)
        slot1_worst_max = np.empty(S, dtype=np.int32)
        slot1_worst_min = np.empty(S, dtype=np.int32)

        for s in range(S):
            if s == dura_idx:
                bm = db_stat_max[0, s]
                wm = db_stat_max[0, s]
                bn = db_stat_min[0, s]
                wn = db_stat_min[0, s]
                for i in range(1, db_count):
                    v = db_stat_max[i, s]
                    if v > bm: bm = v
                    if v < wm: wm = v
                    v = db_stat_min[i, s]
                    if v > bn: bn = v
                    if v < wn: wn = v
            else:
                v = (db_stat_max[0, s] * eff1) // 100
                bm = v
                wm = v
                v = (db_stat_min[0, s] * eff1) // 100
                bn = v
                wn = v
                for i in range(1, db_count):
                    v = (db_stat_max[i, s] * eff1) // 100
                    if v > bm: bm = v
                    if v < wm: wm = v
                    v = (db_stat_min[i, s] * eff1) // 100
                    if v > bn: bn = v
                    if v < wn: wn = v
            slot1_best_max[s] = bm
            slot1_worst_max[s] = wm
            slot1_best_min[s] = bn
            slot1_worst_min[s] = wn

        local_best = init_best_score
        local_best_i0 = -1
        local_best_i1 = -1
        local_searched = 0

        # Per-i0 after-state, needed again for composite UB.
        after_min_arr = np.empty(S, dtype=np.int32)
        after_max_arr = np.empty(S, dtype=np.int32)
        # Per-i1 final-state, needed again for composite leaf check + score.
        final_min_arr = np.empty(S, dtype=np.int32)
        final_max_arr = np.empty(S, dtype=np.int32)

        for i0 in range(db_count):
            # Apply i0: running state after placing first ingredient.
            # Compute UB on final score assuming slot 1 takes the best-possible ingredient.
            ub_score = 0.0
            for s in range(S):
                if s == dura_idx:
                    after_min = base_min_matrix[m, s] + db_stat_min[i0, s]
                    after_max = base_max_matrix[m, s] + db_stat_max[i0, s]
                else:
                    after_min = base_min_matrix[m, s] + (db_stat_min[i0, s] * eff0) // 100
                    after_max = base_max_matrix[m, s] + (db_stat_max[i0, s] * eff0) // 100

                after_min_arr[s] = after_min
                after_max_arr[s] = after_max

                w = weights[s]
                if w > 0.0:
                    ub_max = after_max + slot1_best_max[s]
                    ub_min = after_min + slot1_best_min[s]
                    ub_score += w * (ub_max * 0.99 + ub_min * 0.01)
                elif w < 0.0:
                    lb_max = after_max + slot1_worst_max[s]
                    lb_min = after_min + slot1_worst_min[s]
                    ub_score += w * (lb_max * 0.99 + lb_min * 0.01)

            # Composite UB: per-formula bound on [after_* + slot1_{worst,best}_*] rectangle.
            for c in range(comp_count):
                a = comp_dep_a_proj[c]
                b = comp_dep_b_proj[c]
                w = comp_weight[c]
                if w == 0.0:
                    continue
                f = comp_formula[c]

                a_max_lo = after_max_arr[a] + slot1_worst_max[a]
                a_max_hi = after_max_arr[a] + slot1_best_max[a]
                b_max_lo = after_max_arr[b] + slot1_worst_max[b]
                b_max_hi = after_max_arr[b] + slot1_best_max[b]
                a_min_lo = after_min_arr[a] + slot1_worst_min[a]
                a_min_hi = after_min_arr[a] + slot1_best_min[a]
                b_min_lo = after_min_arr[b] + slot1_worst_min[b]
                b_min_hi = after_min_arr[b] + slot1_best_min[b]

                if f == FORMULA_MUL_DIV_100:
                    pmax_lo, pmax_hi = _product_bounds_div100(a_max_lo, a_max_hi, b_max_lo, b_max_hi)
                    pmin_lo, pmin_hi = _product_bounds_div100(a_min_lo, a_min_hi, b_min_lo, b_min_hi)
                elif f == FORMULA_RAW_TO_PCT:
                    pmax_lo, pmax_hi = _raw_to_pct_bounds(a_max_lo, a_max_hi, b_max_lo, b_max_hi)
                    pmin_lo, pmin_hi = _raw_to_pct_bounds(a_min_lo, a_min_hi, b_min_lo, b_min_hi)
                elif f == FORMULA_EHP:
                    cc = comp_dep_c_proj[c]
                    c_max_lo = after_max_arr[cc] + slot1_worst_max[cc]
                    c_max_hi = after_max_arr[cc] + slot1_best_max[cc]
                    c_min_lo = after_min_arr[cc] + slot1_worst_min[cc]
                    c_min_hi = after_min_arr[cc] + slot1_best_min[cc]
                    pmax_lo, pmax_hi = _ehp_bounds(a_max_lo, a_max_hi, b_max_lo, b_max_hi, c_max_lo, c_max_hi)
                    pmin_lo, pmin_hi = _ehp_bounds(a_min_lo, a_min_hi, b_min_lo, b_min_hi, c_min_lo, c_min_hi)
                else:  # FORMULA_EHPR
                    cc = comp_dep_c_proj[c]
                    dd = comp_dep_d_proj[c]
                    c_max_lo = after_max_arr[cc] + slot1_worst_max[cc]
                    c_max_hi = after_max_arr[cc] + slot1_best_max[cc]
                    d_max_lo = after_max_arr[dd] + slot1_worst_max[dd]
                    d_max_hi = after_max_arr[dd] + slot1_best_max[dd]
                    c_min_lo = after_min_arr[cc] + slot1_worst_min[cc]
                    c_min_hi = after_min_arr[cc] + slot1_best_min[cc]
                    d_min_lo = after_min_arr[dd] + slot1_worst_min[dd]
                    d_min_hi = after_min_arr[dd] + slot1_best_min[dd]
                    pmax_lo, pmax_hi = _ehpr_bounds(
                        a_max_lo, a_max_hi, b_max_lo, b_max_hi,
                        c_max_lo, c_max_hi, d_max_lo, d_max_hi,
                    )
                    pmin_lo, pmin_hi = _ehpr_bounds(
                        a_min_lo, a_min_hi, b_min_lo, b_min_hi,
                        c_min_lo, c_min_hi, d_min_lo, d_min_hi,
                    )

                if w > 0.0:
                    ub_score += w * (pmax_hi * 0.99 + pmin_hi * 0.01)
                else:
                    ub_score += w * (pmax_lo * 0.99 + pmin_lo * 0.01)

            if ub_score <= local_best:
                continue

            # i1 loop — eff == eff constraint drops permutation duplicates.
            start_i1 = i0 if same_eff else 0
            for i1 in range(start_i1, db_count):
                feasible = True
                score = 0.0
                for s in range(S):
                    if s == dura_idx:
                        v_min = base_min_matrix[m, s] + db_stat_min[i0, s] + db_stat_min[i1, s]
                        v_max = base_max_matrix[m, s] + db_stat_max[i0, s] + db_stat_max[i1, s]
                    else:
                        v_min = (base_min_matrix[m, s]
                                 + (db_stat_min[i0, s] * eff0) // 100
                                 + (db_stat_min[i1, s] * eff1) // 100)
                        v_max = (base_max_matrix[m, s]
                                 + (db_stat_max[i0, s] * eff0) // 100
                                 + (db_stat_max[i1, s] * eff1) // 100)

                    final_min_arr[s] = v_min
                    final_max_arr[s] = v_max

                    if has_min_mask[s] and v_max < min_vals[s]:
                        feasible = False
                        break
                    if has_max_mask[s] and v_min > max_vals[s]:
                        feasible = False
                        break

                    score += weights[s] * (v_max * 0.99 + v_min * 0.01)

                if not feasible:
                    continue

                # Composite constraint + score on the finalized k=2 build.
                for c in range(comp_count):
                    a = comp_dep_a_proj[c]
                    b = comp_dep_b_proj[c]
                    f = comp_formula[c]
                    if f == FORMULA_MUL_DIV_100:
                        cmax = (np.int64(final_max_arr[a]) * np.int64(final_max_arr[b])) // 100
                        cmin = (np.int64(final_min_arr[a]) * np.int64(final_min_arr[b])) // 100
                    elif f == FORMULA_RAW_TO_PCT:
                        cmax = _raw_to_pct(final_max_arr[a], final_max_arr[b])
                        cmin = _raw_to_pct(final_min_arr[a], final_min_arr[b])
                    elif f == FORMULA_EHP:
                        cc = comp_dep_c_proj[c]
                        cmin, cmax = _ehp_bounds(
                            final_min_arr[a], final_max_arr[a],
                            final_min_arr[b], final_max_arr[b],
                            final_min_arr[cc], final_max_arr[cc],
                        )
                    else:  # FORMULA_EHPR
                        cc = comp_dep_c_proj[c]
                        dd = comp_dep_d_proj[c]
                        cmin, cmax = _ehpr_bounds(
                            final_min_arr[a], final_max_arr[a],
                            final_min_arr[b], final_max_arr[b],
                            final_min_arr[cc], final_max_arr[cc],
                            final_min_arr[dd], final_max_arr[dd],
                        )
                    if comp_has_min[c] and cmax < comp_min[c]:
                        feasible = False
                        break
                    if comp_has_max[c] and cmin > comp_max[c]:
                        feasible = False
                        break
                    score += comp_weight[c] * (cmax * 0.99 + cmin * 0.01)

                if not feasible:
                    continue

                local_searched += 1
                if score > local_best:
                    local_best = score
                    local_best_i0 = i0
                    local_best_i1 = i1

        best_scores[m] = local_best
        if local_best_i0 >= 0:
            best_solutions[m, 0] = local_best_i0
            best_solutions[m, 1] = local_best_i1
        per_m_searched[m] = local_searched

    # Global reduction
    best_score = -1e18
    best_meta_index = -1
    best_solution_global = np.zeros(2, dtype=np.int32)
    for m in range(M):
        if best_scores[m] > best_score:
            best_score = best_scores[m]
            best_meta_index = m
            best_solution_global[0] = best_solutions[m, 0]
            best_solution_global[1] = best_solutions[m, 1]

    total = 0
    for m in range(M):
        total += per_m_searched[m]

    return best_score, best_meta_index, best_solution_global, total


# ============================================================
# Search One Meta Batch (numba)
# ============================================================

@njit(parallel=True, cache=True)
def search_meta_batch(
    ings_matrix,
    void_count,
    void_eff_matrix,
    base_min_matrix,
    base_max_matrix,
    db_stat_min,
    db_stat_max,
    db_contrib_pos_mask,
    db_contrib_neg_mask,
    db_count,
    dura_idx,
    has_min_mask,
    has_max_mask,
    pos_weight_mask,
    neg_weight_mask,
    min_vals,
    max_vals,
    weights,
    total_searched,
    init_best_score,
    comp_count,
    comp_formula,
    comp_dep_a_proj,
    comp_dep_b_proj,
    comp_dep_c_proj,
    comp_dep_d_proj,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
):
    M = ings_matrix.shape[0]
    k = void_count

    # Per-m output slots; no shared state between iterations.
    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, k), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    for m in prange(M):
        ingredients = np.zeros(k, dtype=np.int32)
        current_min = base_min_matrix[m].copy()
        current_max = base_max_matrix[m].copy()
        best_score_ref = np.array([init_best_score], dtype=np.float64)
        best_solution_local = np.zeros(k, dtype=np.int32)
        local_searched = np.zeros(1, dtype=np.int64)

        # Precompute UB/LB suffix sums for this meta set.
        future_max_ub, future_min_ub, future_max_lb, future_min_lb = \
            _precompute_bounds(
                db_stat_min, db_stat_max, void_eff_matrix[m], k, dura_idx,
            )

        dfs(
            0,
            0,
            k,
            ingredients,
            current_min,
            current_max,
            best_score_ref,
            best_solution_local,
            db_stat_min,
            db_stat_max,
            db_contrib_pos_mask,
            db_contrib_neg_mask,
            db_count,
            void_eff_matrix[m],
            dura_idx,
            has_min_mask,
            has_max_mask,
            pos_weight_mask,
            neg_weight_mask,
            min_vals,
            max_vals,
            weights,
            local_searched,
            future_max_ub,
            future_min_ub,
            future_max_lb,
            future_min_lb,
            comp_count,
            comp_formula,
            comp_dep_a_proj,
            comp_dep_b_proj,
            comp_dep_c_proj,
            comp_dep_d_proj,
            comp_min,
            comp_max,
            comp_has_min,
            comp_has_max,
            comp_weight,
        )

        best_scores[m] = best_score_ref[0]
        for i in range(k):
            best_solutions[m, i] = best_solution_local[i]
        per_m_searched[m] = local_searched[0]

    # Serial reduction — picks the earliest m with the max score (matches
    # the sequential loop's tie-break).
    best_score = -1e18
    best_meta_index = -1
    best_solution_global = np.zeros(k, dtype=np.int32)
    for m in range(M):
        if best_scores[m] > best_score:
            best_score = best_scores[m]
            best_meta_index = m
            for i in range(k):
                best_solution_global[i] = best_solutions[m, i]

    total = 0
    for m in range(M):
        total += per_m_searched[m]
    total_searched[0] += total

    return best_score, best_meta_index, best_solution_global


# ============================================================
# Python Orchestration
# ============================================================

def _dispatch_search(meta_batch, db, query, dura_idx, total_searched, best_score):
    """Run one meta batch through the right kernel based on void_count."""
    if meta_batch.void_count == 1:
        score, meta_index, sol, added = _search_meta_batch_k1(
            meta_batch.void_eff_matrix,
            meta_batch.base_min_matrix,
            meta_batch.base_max_matrix,
            db.stat_min_matrix,
            db.stat_max_matrix,
            db.count,
            dura_idx,
            query.has_min_mask_proj,
            query.has_max_mask_proj,
            query.min_proj,
            query.max_proj,
            query.weights_proj,
            best_score,
            query.comp_count,
            query.comp_formula,
            query.comp_dep_a_proj,
            query.comp_dep_b_proj,
            query.comp_dep_c_proj,
            query.comp_dep_d_proj,
            query.comp_min,
            query.comp_max,
            query.comp_has_min,
            query.comp_has_max,
            query.comp_weight,
        )
        total_searched[0] += added
        return score, meta_index, sol

    if meta_batch.void_count == 2:
        score, meta_index, sol, added = _search_meta_batch_k2(
            meta_batch.void_eff_matrix,
            meta_batch.base_min_matrix,
            meta_batch.base_max_matrix,
            db.stat_min_matrix,
            db.stat_max_matrix,
            db.count,
            dura_idx,
            query.has_min_mask_proj,
            query.has_max_mask_proj,
            query.min_proj,
            query.max_proj,
            query.weights_proj,
            best_score,
            query.comp_count,
            query.comp_formula,
            query.comp_dep_a_proj,
            query.comp_dep_b_proj,
            query.comp_dep_c_proj,
            query.comp_dep_d_proj,
            query.comp_min,
            query.comp_max,
            query.comp_has_min,
            query.comp_has_max,
            query.comp_weight,
        )
        total_searched[0] += added
        return score, meta_index, sol

    return search_meta_batch(
        meta_batch.ings_matrix,
        meta_batch.void_count,
        meta_batch.void_eff_matrix,
        meta_batch.base_min_matrix,
        meta_batch.base_max_matrix,
        db.stat_min_matrix,
        db.stat_max_matrix,
        db.contrib_pos_mask,
        db.contrib_neg_mask,
        db.count,
        dura_idx,
        query.has_min_mask_proj,
        query.has_max_mask_proj,
        query.pos_weight_mask_proj,
        query.neg_weight_mask_proj,
        query.min_proj,
        query.max_proj,
        query.weights_proj,
        total_searched,
        best_score,
        query.comp_count,
        query.comp_formula,
        query.comp_dep_a_proj,
        query.comp_dep_b_proj,
        query.comp_dep_c_proj,
        query.comp_dep_d_proj,
        query.comp_min,
        query.comp_max,
        query.comp_has_min,
        query.comp_has_max,
        query.comp_weight,
    )


def _update_best(meta_batch, score, meta_index, sol, db, best_score, best_full_slots):
    if score > best_score and meta_index != -1:
        best_score = score
        meta_ings = meta_batch.ings_matrix[meta_index]
        full_slots = meta_ings.copy()
        void_perm = meta_batch.void_slots_matrix[meta_index]
        for sorted_depth in range(meta_batch.void_count):
            real_slot = void_perm[sorted_depth]
            db_idx = sol[sorted_depth]
            full_slots[real_slot] = db.json_ids[db_idx]
        best_full_slots = full_slots
    return best_score, best_full_slots


def search(all_meta_sets, db, query):

    best_score = -1e18
    best_full_slots = None
    total_searched = np.array([0], dtype=np.int64)
    total_possibilities = 0

    dura_idx = query.dura_proj_idx

    if dura_idx == -1:
        print("Dura not found in query. Aborting.")
        return None
    elif query.min_proj[dura_idx] < 1:
        print("Min dura should be strictly positive. Aborting.")
        return None

    # Process batches largest-M first: they are the fastest to search (shallow
    # DFS) and provide a tight best_score baseline that the small-M batches
    # (deeper DFS, slower) can use to prune aggressively.
    ordered = sorted(all_meta_sets, key=lambda b: -b.ings_matrix.shape[0])

    for meta_batch in ordered:
        start_time = time()

        if meta_batch.ings_matrix.shape[0] == 0:
            continue

        score, meta_index, sol = _dispatch_search(
            meta_batch, db, query, dura_idx, total_searched, best_score,
        )
        best_score, best_full_slots = _update_best(
            meta_batch, score, meta_index, sol, db, best_score, best_full_slots,
        )

        print(f"meta batch {6-meta_batch.void_count}: {len(meta_batch.ings_matrix)}, time elapsed: {(time()-start_time)*1000:.0f}ms")
        total_possibilities += len(meta_batch.ings_matrix) * db.count**meta_batch.void_count

    print()
    print("SEARCHED FINISHED")
    print(f"Total combinations : {total_possibilities:,}")
    print(f"Total evaluated : {total_searched[0]:,}")
    print(f"Pruning efficiency : {(1-total_searched[0]/total_possibilities)*100:.2f}% skipped")
    print()
    return best_full_slots


# ============================================================
# Pipelined load + search (background loader thread)
# ============================================================

def search_pipelined(
    skill,
    query,
    recipe,
    db,
    max_cull=5,
    culling=True,
    base_path="data/precalc/generic_cull",
):
    """
    Overlap meta-set loading with searching. A background thread loads and
    refines each META_n file in priority order, pushing ready batches onto a
    queue; the main thread searches them as they arrive. Wall-clock = max(load
    time, search time) instead of their sum.
    """
    # Local import avoids a module-level cycle (meta_set_loader imports nothing
    # from search_engine, but search_engine doesn't want the refiner eagerly).
    from data.meta_set_loader import _load_cached_arrays, _refine_batch

    dura_idx = query.dura_proj_idx
    if dura_idx == -1:
        print("Dura not found in query. Aborting.")
        return None
    elif query.min_proj[dura_idx] < 1:
        print("Min dura should be strictly positive. Aborting.")
        return None

    # Priority: META_5 first (cheap search, sets best_score quickly), then
    # descending k for the rest. META_0 is synthetic and trivial.
    priorities = [5, 4, 3, 2, 1, 0]

    q = Queue(maxsize=2)

    def producer():
        try:
            for n in priorities:
                if n == 0:
                    empty_ings = np.full((1, 6), -1, dtype=np.int32)
                    empty_eff = np.full((1, 6), 100, dtype=np.int32)
                    empty_stat_names = np.empty(0, dtype="U1")
                    empty_stat_min = np.zeros((1, 0), dtype=np.int32)
                    empty_stat_max = np.zeros((1, 0), dtype=np.int32)
                    batch = _refine_batch(
                        empty_ings, empty_eff, empty_stat_names,
                        empty_stat_min, empty_stat_max,
                        query, recipe, culling=False,
                    )
                else:
                    use_culling = culling and n <= max_cull
                    ings, eff, stat_names, stat_min, stat_max = \
                        _load_cached_arrays(skill, n, base_path)
                    batch = _refine_batch(
                        ings, eff, stat_names, stat_min, stat_max,
                        query, recipe, culling=use_culling,
                    )
                q.put((n, batch))
        except Exception as e:
            q.put(("ERROR", e))
        finally:
            q.put(None)

    t = Thread(target=producer, daemon=True)
    t.start()

    best_score = -1e18
    best_full_slots = None
    total_searched = np.array([0], dtype=np.int64)
    total_possibilities = 0

    while True:
        item = q.get()
        if item is None:
            break
        tag, payload = item
        if tag == "ERROR":
            t.join()
            raise payload

        n = tag
        meta_batch = payload
        start_time = time()

        if meta_batch.ings_matrix.shape[0] == 0:
            continue

        score, meta_index, sol = _dispatch_search(
            meta_batch, db, query, dura_idx, total_searched, best_score,
        )
        best_score, best_full_slots = _update_best(
            meta_batch, score, meta_index, sol, db, best_score, best_full_slots,
        )

        print(
            f"meta batch {6 - meta_batch.void_count}: "
            f"{len(meta_batch.ings_matrix)}, "
            f"time elapsed: {(time() - start_time) * 1000:.0f}ms"
        )
        total_possibilities += len(meta_batch.ings_matrix) * db.count ** meta_batch.void_count

    t.join()

    print()
    print("SEARCHED FINISHED")
    print(f"Total combinations : {total_possibilities:,}")
    print(f"Total evaluated : {total_searched[0]:,}")
    if total_possibilities > 0:
        print(f"Pruning efficiency : {(1 - total_searched[0] / total_possibilities) * 100:.2f}% skipped")
    print()
    return best_full_slots