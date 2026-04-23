"""
search_engine.py
"""

import numpy as np
from numba import njit, prange
from numba import types
from queue import Queue
from threading import Thread
from time import time


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
)


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

                if has_min_mask[s] and max_v < min_vals[s]:
                    feasible = False
                    break
                if has_max_mask[s] and min_v > max_vals[s]:
                    feasible = False
                    break

                score += weights[s] * (max_v * 0.99 + min_v * 0.01)

            if feasible:
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

                w = weights[s]
                if w > 0.0:
                    ub_max = after_max + slot1_best_max[s]
                    ub_min = after_min + slot1_best_min[s]
                    ub_score += w * (ub_max * 0.99 + ub_min * 0.01)
                elif w < 0.0:
                    lb_max = after_max + slot1_worst_max[s]
                    lb_min = after_min + slot1_worst_min[s]
                    ub_score += w * (lb_max * 0.99 + lb_min * 0.01)

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

                    if has_min_mask[s] and v_max < min_vals[s]:
                        feasible = False
                        break
                    if has_max_mask[s] and v_min > max_vals[s]:
                        feasible = False
                        break

                    score += weights[s] * (v_max * 0.99 + v_min * 0.01)

                if feasible:
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