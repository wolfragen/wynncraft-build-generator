"""
pareto_search.py

Pareto-precalc search mode. Generates a frontier of "promising" recipes
(ingredient combinations) over K user-defined axes. Each axis is a DSL
program (see `dsl/`); the search visits all META_N batches for the
target recipe, evaluates the K axes per leaf, and keeps every recipe
that achieves at least θ of the current best on at least one axis.

This module is INTENTIONALLY SEPARATE from `core/search_engine.py`. The
existing scalar-score DFS is not modified. We share only:
    - data.meta_set_loader   (META_N batch loading + cache)
    - data.ingredient_db     (compact ingredient stat matrices)
    - data.recipe            (recipe scaling)
    - dsl                    (axis programs)
    - query.query            (Query parsing — but with the new
                              ingredient_filter_stats / hard_constraints
                              / axes fields handled here).

Output
------
A list of records, each with:
    ingredients : tuple[int]    — ingredient IDs (matches the search slot
                                  order)
    axis_lb     : tuple[float]  — per-axis lower bound over rolls
    axis_ub     : tuple[float]  — per-axis upper bound over rolls

Streaming gate
--------------
During DFS we maintain `best_per_axis[K]` (monotone-increasing) and
threshold-gate every candidate recipe at the leaf:

    keep iff   max_k(axis_ub_R[k] / best_per_axis[k]) >= θ

This lets a recipe survive on the strength of a single axis (the
"specialization" use case the user asked for). After DFS we re-apply
the gate against the FINAL `best_per_axis` and run a strict Pareto cull
over (axis_ub) vectors.

B&B pruning
-----------
At each interior DFS node we compute an admissible UB on each axis
using the DSL's `eval_program_bounds`. If for every axis k the UB
drops below θ * best_per_axis[k], the subtree can never produce a
recipe that passes the gate — we prune. This requires `best_per_axis`
to be visible to the kernel; we pass it as a mutable scratch array.
"""

import numpy as np
from numba import njit, types

from data.skillpoint_lookup import SKP_HEADLINE_PCT, SKP_ELEMENT_PCT
from dsl.evaluator import eval_program, eval_program_bounds
from core.search_engine import _precompute_bounds


# ============================================================
# Numba-accessible per-axis "program block" layout
# ============================================================
# K axis programs are concatenated into flat arrays so the kernel can
# iterate them. For axis k:
#     ops      = axis_op_codes[axis_op_offset[k] : axis_op_offset[k] + axis_op_count[k]]
#     args     = axis_op_args [...same offsets...]
#     consts   = axis_consts  [axis_const_offset[k] : axis_const_offset[k] + axis_const_count[k]]
# The constant-pool indices stored in each axis's LIT instructions are
# RELATIVE to that axis's slice of `axis_consts` (so we just index with
# axis_consts[axis_const_offset[k] + lit_idx]).
#
# `axis_is_monotone[k]` and `axis_max_stack[k]` come from each Program's
# compile-time metadata.


def pack_axis_programs(programs):
    """Flatten K compiled Programs into the 7 arrays the numba kernel
    consumes. Returns a dict so the caller can splat into the kernel
    call site without keyword churn."""
    K = len(programs)
    if K == 0:
        raise ValueError("Need at least one axis program")

    op_offsets   = np.zeros(K, dtype=np.int32)
    op_counts    = np.zeros(K, dtype=np.int32)
    cst_offsets  = np.zeros(K, dtype=np.int32)
    cst_counts   = np.zeros(K, dtype=np.int32)
    is_monotone  = np.zeros(K, dtype=np.bool_)
    max_stack    = np.zeros(K, dtype=np.int32)

    op_codes_l = []
    op_args_l  = []
    consts_l   = []

    op_acc = 0
    cst_acc = 0
    for k, prog in enumerate(programs):
        op_offsets[k]  = op_acc
        op_counts[k]   = prog.op_codes.shape[0]
        cst_offsets[k] = cst_acc
        cst_counts[k]  = prog.consts.shape[0]
        is_monotone[k] = prog.is_monotone
        max_stack[k]   = prog.max_stack
        op_codes_l.append(prog.op_codes)
        op_args_l.append(prog.op_args)
        consts_l.append(prog.consts)
        op_acc  += prog.op_codes.shape[0]
        cst_acc += prog.consts.shape[0]

    op_codes = np.concatenate(op_codes_l).astype(np.int32, copy=False)
    op_args  = np.concatenate(op_args_l, axis=0).astype(np.int32, copy=False)
    consts   = np.concatenate(consts_l).astype(np.float64, copy=False)

    return dict(
        K=K,
        axis_op_codes=op_codes,
        axis_op_args=op_args,
        axis_consts=consts,
        axis_op_offset=op_offsets,
        axis_op_count=op_counts,
        axis_const_offset=cst_offsets,
        axis_const_count=cst_counts,
        axis_is_monotone=is_monotone,
        axis_max_stack=max_stack,
        max_stack_global=int(max_stack.max()) if K else 1,
    )


# ============================================================
# Per-axis evaluation helpers (numba-inlinable wrappers)
# ============================================================

@njit(cache=True, fastmath=True, inline="always")
def _eval_one_axis(
    k_axis,
    axis_op_codes, axis_op_args, axis_consts,
    axis_op_offset, axis_op_count, axis_const_offset,
    stats, ctx, scratch,
    skp_headline, skp_element,
):
    """Evaluate axis `k_axis` at a single stat vector. Returns the value.

    We call eval_program with sliced views — numba supports array slicing
    with literal types since 0.55. Slices are no-copy."""
    o0 = axis_op_offset[k_axis]
    o1 = o0 + axis_op_count[k_axis]
    c0 = axis_const_offset[k_axis]
    return eval_program(
        axis_op_codes[o0:o1],
        axis_op_args [o0:o1],
        axis_consts  [c0:],            # the axis only references its own slice; idx in
                                        # LIT instructions is local-to-axis (cst_offset[k] + idx)
                                        # so we pass the slice from cst_offset[k] onward.
        stats, ctx, scratch,
        skp_headline, skp_element,
    )


@njit(cache=True, fastmath=True, inline="always")
def _eval_one_axis_bounds(
    k_axis,
    axis_op_codes, axis_op_args, axis_consts,
    axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
    stats_lb, stats_ub, ctx, scratch_a, scratch_b,
    skp_headline, skp_element,
):
    """Returns (lb, ub) for axis k_axis over [stats_lb, stats_ub]."""
    o0 = axis_op_offset[k_axis]
    o1 = o0 + axis_op_count[k_axis]
    c0 = axis_const_offset[k_axis]
    return eval_program_bounds(
        axis_op_codes[o0:o1],
        axis_op_args [o0:o1],
        axis_consts  [c0:],
        stats_lb, stats_ub, ctx,
        scratch_a, scratch_b,
        axis_is_monotone[k_axis],
        skp_headline, skp_element,
    )


# ============================================================
# Pareto-mode DFS kernel
# ============================================================
# Mirrors `dfs` in core/search_engine.py but with K-axis evaluation and
# threshold gate at leaves. No scoring, no composites, no SP-cap stuff
# — those belong to the scalar-score path. Hard min/max constraints on
# stats still apply via has_min_mask / has_max_mask.

@njit(cache=False)
def pareto_dfs(
    depth, start_index, k,
    ingredients,                  # int32[k]
    current_min, current_max,     # int32[S]
    db_stat_min, db_stat_max,     # int16[N, S]
    db_count,
    meta_void_eff,                # int32[k]
    dura_idx,
    has_min_mask, has_max_mask,   # bool[S]
    min_vals, max_vals,           # int32[S]
    round_offset,                 # int32[S]
    future_max_ub, future_min_ub, # int32[k+1, S]
    future_max_lb, future_min_lb,
    # Axes
    K_axes,
    axis_op_codes, axis_op_args, axis_consts,
    axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
    # Eval scratch buffers (sized to global max_stack)
    scratch_a, scratch_b,
    # Reusable float64 stat copies (kernel writes to these per leaf/node)
    stats_lb_f, stats_ub_f,
    # Build ctx + SKP tables
    build_ctx, skp_headline, skp_element,
    # Threshold gate state
    threshold,                    # float64
    best_per_axis,                # float64[K] mutable
    # Output buffer (Python pre-allocates a generous size)
    out_recipes,                  # int32[MAX, k]
    out_axes_lb,                  # float64[MAX, K]
    out_axes_ub,                  # float64[MAX, K]
    out_count,                    # int64[1]
    out_max,                      # int64
    # Pruning inputs
    useful_pos_eff, useful_neg_eff,
    total_searched,
):
    # ----------------------------------------------------------
    # Leaf
    # ----------------------------------------------------------
    if depth == k:
        total_searched[0] += 1

        # Hard constraint check.
        S = current_min.shape[0]
        for s in range(S):
            if has_min_mask[s] and current_max[s] < min_vals[s]:
                return
            if has_max_mask[s] and current_min[s] > max_vals[s]:
                return

        # Axis evaluation: (lb, ub) per axis using the DSL bounds eval
        # over the rolling box [current_min, current_max].
        for s in range(S):
            stats_lb_f[s] = float(current_min[s])
            stats_ub_f[s] = float(current_max[s])

        # First pass: compute axis_ub for every axis (we need them for
        # the threshold gate).
        passes_gate = False
        for ka in range(K_axes):
            lb, ub = _eval_one_axis_bounds(
                ka,
                axis_op_codes, axis_op_args, axis_consts,
                axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
                stats_lb_f, stats_ub_f, build_ctx,
                scratch_a, scratch_b,
                skp_headline, skp_element,
            )
            # Update best.
            if ub > best_per_axis[ka]:
                best_per_axis[ka] = ub
            # Threshold check on this axis (any axis passing keeps R).
            ref = best_per_axis[ka]
            if ref > 0.0:
                if ub >= threshold * ref:
                    passes_gate = True
            else:
                # Both ub and ref are 0 (or negative): degenerate axis,
                # treat as passing — caller can post-filter if desired.
                passes_gate = True

        if not passes_gate:
            return

        # Second pass: store the lb/ub vectors into the output buffer.
        idx = out_count[0]
        if idx >= out_max:
            # Buffer overflow — caller will detect this and raise.
            out_count[0] = -1
            return
        for ka in range(K_axes):
            lb, ub = _eval_one_axis_bounds(
                ka,
                axis_op_codes, axis_op_args, axis_consts,
                axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
                stats_lb_f, stats_ub_f, build_ctx,
                scratch_a, scratch_b,
                skp_headline, skp_element,
            )
            out_axes_lb[idx, ka] = lb
            out_axes_ub[idx, ka] = ub
        for j in range(k):
            out_recipes[idx, j] = ingredients[j]
        out_count[0] = idx + 1
        return

    # ----------------------------------------------------------
    # Interior node — branch over ingredients in this slot
    # ----------------------------------------------------------
    eff = meta_void_eff[depth]
    eff_is_positive = eff > 0
    eff_pos = eff >= 0
    next_depth = depth + 1
    has_next_slot = next_depth < k
    same_eff_next = has_next_slot and (eff == meta_void_eff[next_depth])

    if eff_is_positive:
        useful_for_eff = useful_pos_eff
    else:
        useful_for_eff = useful_neg_eff

    S = current_min.shape[0]

    for i in range(start_index, db_count):
        if not useful_for_eff[i]:
            continue

        # Dura pruning (matches existing DFS).
        if current_max[dura_idx] + db_stat_min[i, dura_idx] < min_vals[dura_idx]:
            continue

        ingredients[depth] = i

        # Apply this ingredient's contribution.
        for s in range(S):
            if s == dura_idx:
                current_min[s] += db_stat_min[i, s]
                current_max[s] += db_stat_max[i, s]
            else:
                ro = round_offset[s]
                if eff_pos:
                    current_min[s] += (db_stat_min[i, s] * eff + ro) // 100
                    current_max[s] += (db_stat_max[i, s] * eff + ro) // 100
                else:
                    current_min[s] += (db_stat_max[i, s] * eff + ro) // 100
                    current_max[s] += (db_stat_min[i, s] * eff + ro) // 100

        # Hard feasibility check using future bounds.
        feasible = True
        for s in range(S):
            if has_min_mask[s] and current_max[s] + future_max_ub[next_depth, s] < min_vals[s]:
                feasible = False
                break
            if has_max_mask[s] and current_min[s] + future_min_lb[next_depth, s] > max_vals[s]:
                feasible = False
                break

        if not feasible:
            for s in range(S):
                if s == dura_idx:
                    current_min[s] -= db_stat_min[i, s]
                    current_max[s] -= db_stat_max[i, s]
                else:
                    ro = round_offset[s]
                    if eff_pos:
                        current_min[s] -= (db_stat_min[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_max[i, s] * eff + ro) // 100
                    else:
                        current_min[s] -= (db_stat_max[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_min[i, s] * eff + ro) // 100
            continue

        # Axis-UB pruning: build optimistic stats_ub (current_max + future_max_ub)
        # and stats_lb (current_min + future_min_lb), evaluate UB of each axis,
        # cut if NO axis can possibly reach threshold * best_per_axis.
        for s in range(S):
            stats_lb_f[s] = float(current_min[s] + future_min_lb[next_depth, s])
            stats_ub_f[s] = float(current_max[s] + future_max_ub[next_depth, s])

        any_axis_promising = False
        for ka in range(K_axes):
            _, ub = _eval_one_axis_bounds(
                ka,
                axis_op_codes, axis_op_args, axis_consts,
                axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
                stats_lb_f, stats_ub_f, build_ctx,
                scratch_a, scratch_b,
                skp_headline, skp_element,
            )
            ref = best_per_axis[ka]
            if ref > 0.0:
                if ub >= threshold * ref:
                    any_axis_promising = True
                    break
            else:
                any_axis_promising = True
                break

        if not any_axis_promising:
            for s in range(S):
                if s == dura_idx:
                    current_min[s] -= db_stat_min[i, s]
                    current_max[s] -= db_stat_max[i, s]
                else:
                    ro = round_offset[s]
                    if eff_pos:
                        current_min[s] -= (db_stat_min[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_max[i, s] * eff + ro) // 100
                    else:
                        current_min[s] -= (db_stat_max[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_min[i, s] * eff + ro) // 100
            continue

        # Permutation kill (mirrors search_engine.dfs).
        next_start = 0
        if k == 6 or same_eff_next:
            next_start = i

        pareto_dfs(
            depth + 1, next_start, k,
            ingredients,
            current_min, current_max,
            db_stat_min, db_stat_max, db_count,
            meta_void_eff, dura_idx,
            has_min_mask, has_max_mask,
            min_vals, max_vals, round_offset,
            future_max_ub, future_min_ub, future_max_lb, future_min_lb,
            K_axes,
            axis_op_codes, axis_op_args, axis_consts,
            axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
            scratch_a, scratch_b,
            stats_lb_f, stats_ub_f,
            build_ctx, skp_headline, skp_element,
            threshold, best_per_axis,
            out_recipes, out_axes_lb, out_axes_ub, out_count, out_max,
            useful_pos_eff, useful_neg_eff,
            total_searched,
        )

        # Undo.
        for s in range(S):
            if s == dura_idx:
                current_min[s] -= db_stat_min[i, s]
                current_max[s] -= db_stat_max[i, s]
            else:
                ro = round_offset[s]
                if eff_pos:
                    current_min[s] -= (db_stat_min[i, s] * eff + ro) // 100
                    current_max[s] -= (db_stat_max[i, s] * eff + ro) // 100
                else:
                    current_min[s] -= (db_stat_max[i, s] * eff + ro) // 100
                    current_max[s] -= (db_stat_min[i, s] * eff + ro) // 100


# ============================================================
# Per-(ingredient, eff sign) "useful" mask
# ============================================================
# For pareto mode every active stat is potentially useful (it may feed
# an axis or a hard constraint). Conservative: include ingredient i in
# `useful_pos_eff` iff it has ANY positive contribution on ANY active
# stat (and similarly for neg). Looser than the scalar-search mask but
# correct — never kills a recipe that could be promising.

@njit(cache=True)
def _compute_useful_masks_pareto(db_contrib_pos_mask, db_contrib_neg_mask):
    N = db_contrib_pos_mask.shape[0]
    S = db_contrib_pos_mask.shape[1]
    useful_pos = np.zeros(N, dtype=np.bool_)
    useful_neg = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        for s in range(S):
            if db_contrib_pos_mask[i, s] or db_contrib_neg_mask[i, s]:
                useful_pos[i] = True
                useful_neg[i] = True
                break
    return useful_pos, useful_neg


# ============================================================
# Per-meta-batch wrapper
# ============================================================
# One DFS call per meta-set in the batch. Sequential (no prange) for V1
# — parallelization across meta-sets needs per-thread best_per_axis +
# per-thread output buffers + a final merge step, which we'll add once
# the sequential path is validated.

@njit(cache=False)
def search_meta_batch_pareto(
    ings_matrix,                  # int32[M, k] — fixed-slot ingredient IDs
    void_count,                   # = k
    void_eff_matrix,              # int32[M, k]
    base_min_matrix, base_max_matrix,  # int32[M, S] — fixed-slot contributions
    db_stat_min, db_stat_max,
    db_contrib_pos_mask, db_contrib_neg_mask,
    db_count,
    dura_idx,
    has_min_mask, has_max_mask,
    min_vals, max_vals,
    round_offset,
    K_axes,
    axis_op_codes, axis_op_args, axis_consts,
    axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
    max_stack_global,
    build_ctx, skp_headline, skp_element,
    threshold,
    best_per_axis,
    out_recipes,
    out_axes_lb,
    out_axes_ub,
    out_count,
    out_max,
    out_meta_indices,             # int32[MAX] — which m produced each kept recipe
    total_searched,
):
    M = ings_matrix.shape[0]
    k = void_count
    S = base_min_matrix.shape[1]

    useful_pos_eff, useful_neg_eff = _compute_useful_masks_pareto(
        db_contrib_pos_mask, db_contrib_neg_mask,
    )

    # Bounds tensor reused across m's.
    bounds_buf = np.zeros((4, k + 1, S), dtype=np.int32)
    # Float64 stat scratch buffers (per call to pareto_dfs).
    stats_lb_f = np.zeros(S, dtype=np.float64)
    stats_ub_f = np.zeros(S, dtype=np.float64)
    scratch_a = np.zeros(max_stack_global, dtype=np.float64)
    scratch_b = np.zeros(max_stack_global, dtype=np.float64)
    ingredients_buf = np.zeros(k, dtype=np.int32)

    for m in range(M):
        if k == 0:
            # Degenerate META_0 — no void slots, the "recipe" is just the
            # fixed contribution. Evaluate axes once at the [base_min, base_max]
            # box and apply the gate.
            for s in range(S):
                if has_min_mask[s] and base_max_matrix[m, s] < min_vals[s]:
                    break
                if has_max_mask[s] and base_min_matrix[m, s] > max_vals[s]:
                    break
            else:
                for s in range(S):
                    stats_lb_f[s] = float(base_min_matrix[m, s])
                    stats_ub_f[s] = float(base_max_matrix[m, s])
                passes = False
                for ka in range(K_axes):
                    lb, ub = _eval_one_axis_bounds(
                        ka,
                        axis_op_codes, axis_op_args, axis_consts,
                        axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
                        stats_lb_f, stats_ub_f, build_ctx,
                        scratch_a, scratch_b,
                        skp_headline, skp_element,
                    )
                    if ub > best_per_axis[ka]:
                        best_per_axis[ka] = ub
                    ref = best_per_axis[ka]
                    if ref <= 0.0 or ub >= threshold * ref:
                        passes = True
                if passes:
                    idx = out_count[0]
                    if idx >= out_max:
                        out_count[0] = -1
                        return
                    for ka in range(K_axes):
                        lb, ub = _eval_one_axis_bounds(
                            ka,
                            axis_op_codes, axis_op_args, axis_consts,
                            axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
                            stats_lb_f, stats_ub_f, build_ctx,
                            scratch_a, scratch_b,
                            skp_headline, skp_element,
                        )
                        out_axes_lb[idx, ka] = lb
                        out_axes_ub[idx, ka] = ub
                    out_meta_indices[idx] = m
                    out_count[0] = idx + 1
                total_searched[0] += 1
            continue

        current_min = base_min_matrix[m].copy()
        current_max = base_max_matrix[m].copy()

        future_max_ub = bounds_buf[0]
        future_min_ub = bounds_buf[1]
        future_max_lb = bounds_buf[2]
        future_min_lb = bounds_buf[3]
        _precompute_bounds(
            db_stat_min, db_stat_max, void_eff_matrix[m], k, dura_idx, round_offset,
            future_max_ub, future_min_ub, future_max_lb, future_min_lb,
        )

        prev_count = out_count[0]
        pareto_dfs(
            0, 0, k,
            ingredients_buf,
            current_min, current_max,
            db_stat_min, db_stat_max, db_count,
            void_eff_matrix[m], dura_idx,
            has_min_mask, has_max_mask,
            min_vals, max_vals, round_offset,
            future_max_ub, future_min_ub, future_max_lb, future_min_lb,
            K_axes,
            axis_op_codes, axis_op_args, axis_consts,
            axis_op_offset, axis_op_count, axis_const_offset, axis_is_monotone,
            scratch_a, scratch_b,
            stats_lb_f, stats_ub_f,
            build_ctx, skp_headline, skp_element,
            threshold, best_per_axis,
            out_recipes, out_axes_lb, out_axes_ub, out_count, out_max,
            useful_pos_eff, useful_neg_eff,
            total_searched,
        )

        # If overflow happened we propagate the sentinel and bail.
        if out_count[0] < 0:
            return

        # Tag the meta-set index on every recipe produced by this m.
        new_count = out_count[0]
        for i in range(prev_count, new_count):
            out_meta_indices[i] = m


# ============================================================
# Python-side orchestrator
# ============================================================

def _strict_pareto_cull(axes_ub):
    """Strict Pareto cull on (axes_ub) vectors. Returns a boolean mask
    `keep[N]` where keep[i] is True iff no other row j has axes_ub[j, k]
    >= axes_ub[i, k] for all k, with at least one strict inequality.
    O(N^2). Caller can pre-filter to a smaller candidate pool first if
    this becomes a bottleneck."""
    N = axes_ub.shape[0]
    keep = np.ones(N, dtype=np.bool_)
    for i in range(N):
        if not keep[i]:
            continue
        ai = axes_ub[i]
        for j in range(N):
            if j == i or not keep[j]:
                continue
            aj = axes_ub[j]
            # Does j dominate i? aj >= ai everywhere, strict somewhere.
            dominates = True
            strict = False
            for k in range(ai.shape[0]):
                if aj[k] < ai[k]:
                    dominates = False
                    break
                if aj[k] > ai[k]:
                    strict = True
            if dominates and strict:
                keep[i] = False
                break
    return keep


def run_pareto_search(
    skill,
    item_type,
    recipe_lvl,        # tuple (lvl_min, lvl_max)
    tier,
    query,             # built via build_query — its proj_stats_idx etc. drive STAT() resolution
    recipe,            # built via build_recipe
    db,                # IngredientDB
    axes,              # list of (name, Expr)  — Expr is a DSL builder tree
    threshold,         # float, e.g. 0.9
    out_buffer_size=2_000_000,
    base_path="data/precalc/generic_cull",
    verbose=True,
):
    """Drive the Pareto-precalc search across all META_N batches for the
    target recipe. Returns a dict with the kept recipes' (meta_index_per_n,
    void_choices, axes_lb, axes_ub) plus axis names and best_per_axis.
    """
    from time import time
    from data.meta_set_loader import _refine_batch, _load_cached_arrays
    from dsl.builder import compile_program
    from dsl.evaluator import SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64
    from query.query import BUILD_CTX_WD_N as _BCTX_WD_N

    # ----- compile axes -----
    axis_names = [n for (n, _) in axes]
    programs = [compile_program(expr, query) for (_, expr) in axes]
    packed = pack_axis_programs(programs)
    K = packed["K"]
    if verbose:
        print(f"Compiled {K} axes:")
        for i, (n, prog) in enumerate(zip(axis_names, programs)):
            print(f"  [{i}] {n!r}: {prog.op_codes.shape[0]} ops, "
                  f"{prog.consts.shape[0]} consts, "
                  f"max_stack={prog.max_stack}, monotone={prog.is_monotone}")

    # ----- weapon_dam ctx overlay (mirrors search_pipelined) -----
    if recipe.weapon_dam_neutral and (recipe.weapon_dam_neutral[0]
                                      or recipe.weapon_dam_neutral[1]):
        query.build_ctx[_BCTX_WD_N] = (recipe.weapon_dam_neutral[0]
                                       + recipe.weapon_dam_neutral[1]) / 2.0

    dura_idx = query.dura_proj_idx
    if dura_idx == -1:
        raise ValueError("Pareto search requires durability or duration to be active in the query")

    # ----- pre-allocated outputs -----
    # Worst case k=6 (META_6 = no fixed slots). Allocate for k=6 since
    # smaller k just leaves trailing zeros.
    out_recipes = np.full((out_buffer_size, 6), -1, dtype=np.int32)
    out_axes_lb = np.zeros((out_buffer_size, K), dtype=np.float64)
    out_axes_ub = np.zeros((out_buffer_size, K), dtype=np.float64)
    out_meta_indices = np.full(out_buffer_size, -1, dtype=np.int32)
    out_count = np.zeros(1, dtype=np.int64)
    out_max = out_buffer_size

    best_per_axis = np.zeros(K, dtype=np.float64)
    total_searched = np.zeros(1, dtype=np.int64)

    # Track the meta_n source per kept recipe (for reconstruction). The
    # `out_meta_indices` is local to a batch — we need a global identifier
    # too. Stash this in a parallel array post-batch.
    n_per_kept = np.full(out_buffer_size, -1, dtype=np.int8)

    # Per-meta-set ingredient_id matrix for each batch, kept so we can
    # rebuild full slot lists after the search.
    batch_ings_per_n = {}     # n -> int32[M, k]
    batch_void_count_per_n = {}

    # Process META_n in priority order: largest n first (more void slots
    # → highest impact on best_per_axis if any axis amplifies).
    priorities = [5, 4, 3, 2, 1, 0]
    t_total = time()
    for n in priorities:
        t_n = time()

        if n == 0:
            # No precalc file for META_0 (it's the trivial "no void slots"
            # row). Construct it inline.
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
            try:
                cached = _load_cached_arrays(skill, n, base_path)
            except FileNotFoundError:
                if verbose:
                    print(f"META_{n}: missing cache (skipped)")
                continue
            ings_arr, eff_arr, stat_names, stat_min, stat_max = cached
            batch = _refine_batch(
                ings_arr, eff_arr, stat_names, stat_min, stat_max,
                query, recipe, culling=False,
            )

        M = batch.ings_matrix.shape[0]
        if M == 0:
            if verbose:
                print(f"META_{n}: 0 rows after refine")
            continue

        prev_count = int(out_count[0])
        search_meta_batch_pareto(
            batch.ings_matrix,
            batch.void_count,
            batch.void_eff_matrix if batch.void_count > 0
                else np.zeros((M, 0), dtype=np.int32),
            batch.base_min_matrix,
            batch.base_max_matrix,
            db.stat_min_matrix, db.stat_max_matrix,
            db.contrib_pos_mask, db.contrib_neg_mask,
            db.count, dura_idx,
            query.has_min_mask_proj, query.has_max_mask_proj,
            query.min_proj, query.max_proj,
            query.round_offset_proj,
            K,
            packed["axis_op_codes"], packed["axis_op_args"], packed["axis_consts"],
            packed["axis_op_offset"], packed["axis_op_count"], packed["axis_const_offset"],
            packed["axis_is_monotone"],
            packed["max_stack_global"],
            query.build_ctx,
            SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64,
            float(threshold),
            best_per_axis,
            out_recipes,
            out_axes_lb,
            out_axes_ub,
            out_count,
            np.int64(out_max),
            out_meta_indices,
            total_searched,
        )

        if int(out_count[0]) < 0:
            raise RuntimeError(
                f"Pareto search out-buffer overflow during META_{n}. "
                f"Increase out_buffer_size (current={out_buffer_size}) "
                f"or tighten threshold (current={threshold})."
            )

        new_count = int(out_count[0])
        # Tag meta_n on the new entries.
        for i in range(prev_count, new_count):
            n_per_kept[i] = n
        # Save ings_matrix for later reconstruction. Each n has its own
        # M × k layout; we need both to build full slot lists.
        batch_ings_per_n[n] = batch.ings_matrix.copy()
        batch_void_count_per_n[n] = batch.void_count

        if verbose:
            elapsed = time() - t_n
            print(f"META_{n}: M={M:,}, kept {new_count - prev_count:,} "
                  f"recipes (running total {new_count:,}), "
                  f"best_per_axis={list(best_per_axis)}, "
                  f"elapsed={elapsed:.1f}s")

    total = int(out_count[0])
    if verbose:
        print(f"\nDFS complete: {total:,} kept recipes, "
              f"{int(total_searched[0]):,} leaves searched, "
              f"wall={time() - t_total:.1f}s")
        print(f"Final best_per_axis: {list(best_per_axis)}")

    # ----- post-pass 1: re-validate against final best_per_axis -----
    if total > 0:
        ub = out_axes_ub[:total]
        lb = out_axes_lb[:total]
        rec = out_recipes[:total]
        meta_idx = out_meta_indices[:total]
        n_kept = n_per_kept[:total]

        # max_k(ub[k] / best[k]) >= θ — vectorized.
        # Avoid div-by-zero on degenerate axes (best=0 → keep).
        safe_best = np.where(best_per_axis > 0, best_per_axis, 1.0)
        ratios = ub / safe_best  # (N, K)
        max_ratio = ratios.max(axis=1)
        # For degenerate axes (best=0), ratio is meaningless; treat as
        # passing (axis won't differentiate anyway).
        keep_thresh = (max_ratio >= threshold) | (best_per_axis.max() == 0)

        ub = ub[keep_thresh]; lb = lb[keep_thresh]
        rec = rec[keep_thresh]; meta_idx = meta_idx[keep_thresh]
        n_kept = n_kept[keep_thresh]
        if verbose:
            print(f"Post-pass threshold ({threshold}): {keep_thresh.sum():,} of {total:,} survive")

        # ----- post-pass 2: strict Pareto on UB vectors -----
        keep_pareto = _strict_pareto_cull(ub)
        ub = ub[keep_pareto]; lb = lb[keep_pareto]
        rec = rec[keep_pareto]; meta_idx = meta_idx[keep_pareto]
        n_kept = n_kept[keep_pareto]
        if verbose:
            print(f"Post-pass strict Pareto: {keep_pareto.sum():,} survive")
    else:
        ub = np.zeros((0, K))
        lb = np.zeros((0, K))
        rec = np.zeros((0, 6), dtype=np.int32)
        meta_idx = np.zeros(0, dtype=np.int32)
        n_kept = np.zeros(0, dtype=np.int8)

    return {
        "axis_names": axis_names,
        "best_per_axis": best_per_axis.copy(),
        "axes_lb": lb,
        "axes_ub": ub,
        "void_choices": rec,           # (N, k) — DB indices
        "meta_index": meta_idx,        # (N,)  — local to its META_n batch
        "meta_n": n_kept,              # (N,)  — which META_n
        "batch_ings_per_n": batch_ings_per_n,  # n -> ings_matrix[M, k]
        "batch_void_count_per_n": batch_void_count_per_n,
        "threshold": threshold,
        "total_searched": int(total_searched[0]),
    }


# ============================================================
# Serialization
# ============================================================

def save_frontier(result, db, path):
    """Serialize a `run_pareto_search` result to a JSON file. Each kept
    recipe is dumped with full ingredient IDs (composing the meta-set's
    fixed slots with the DFS-chosen voids), the per-axis (lb, ub), and
    the source META_n / meta_index for traceability."""
    import json

    axis_names = result["axis_names"]
    n_keep = result["axes_ub"].shape[0]

    recipes_out = []
    for i in range(n_keep):
        m_n = int(result["meta_n"][i])
        m_idx = int(result["meta_index"][i])
        void_db = result["void_choices"][i]   # (6,) with -1 fill
        ings_matrix = result["batch_ings_per_n"].get(m_n)
        # ings_matrix[m, j] is either an ingredient_id (>=0) for fixed
        # slots or -1 for void slots. Walk and substitute -1's with the
        # next void choice.
        if ings_matrix is None:
            full = []
        else:
            full = ings_matrix[m_idx].tolist()
        v_iter = (int(x) for x in void_db if x >= 0)
        full = [
            int(db.json_ids[next(v_iter)]) if x < 0 else int(x)
            for x in full
        ]
        recipes_out.append({
            "ingredients": full,
            "axes_lb": [float(v) for v in result["axes_lb"][i]],
            "axes_ub": [float(v) for v in result["axes_ub"][i]],
            "meta_n": m_n,
            "meta_index": m_idx,
        })

    payload = {
        "axis_names": axis_names,
        "threshold": result["threshold"],
        "best_per_axis": [float(v) for v in result["best_per_axis"]],
        "total_searched": result["total_searched"],
        "recipes": recipes_out,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
