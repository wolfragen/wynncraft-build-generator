"""
Pre-trigger JIT compilation of every runtime numba kernel so the first real
call is not penalized. Safe to call multiple times (subsequent calls are
instant since the compiled code is already cached in-process).

Call `warm_numba()` once at the start of `main()`.
"""
import numpy as np

from data import meta_set_loader as msl
from data.stats import (
    FORMULA_MUL_DIV_100, FORMULA_RAW_TO_PCT, FORMULA_EHP, FORMULA_EHPR,
    FORMULA_SPELL_DAMAGE_BASE,
)
from data.spells import SPELL_COUNT
from query import ingredient_filter as ingf
from query.query import BUILD_CTX_SIZE
from core import search_engine as se


def warm_numba():
    _warm_meta_set_loader()
    _warm_ingredient_filter()
    _warm_search_engine()


def _warm_meta_set_loader():
    # Small 2-row x 3-col matrix: 1 void-eff column + 2 stat columns.
    mat = np.zeros((2, 3), dtype=np.int32)
    # `_build_cull_matrix` reads lower_better_proj as a bool[num_stats] array;
    # the per-column bool used by the comparator has the void-eff offset
    # baked in (length = num_cols).
    lower_better_proj = np.zeros(1, dtype=np.bool_)
    is_req_col = np.zeros(3, dtype=np.bool_)

    msl.compare_vectors(mat[0], mat[1], 1, is_req_col, -1)
    msl.pareto_filter(mat, 1, is_req_col, -1)

    is_kept = np.ones(2, dtype=np.bool_)
    msl._intra_block_cull(0, 2, mat, 1, is_req_col, -1, is_kept)

    survivors = np.array([0], dtype=np.int64)
    is_kept = np.ones(2, dtype=np.bool_)
    msl._parallel_check_block(1, 2, survivors, 1, mat, 1, is_req_col, -1, is_kept)

    base_min = np.zeros((2, 1), dtype=np.int32)
    base_max = np.zeros((2, 1), dtype=np.int32)
    void_eff = np.zeros((2, 1), dtype=np.int32)
    msl._build_cull_matrix(base_min, base_max, void_eff, 1, lower_better_proj)

    msl.pareto_filter_block(mat, 1, is_req_col, -1, block_size=1)


def _warm_ingredient_filter():
    mat_min = np.zeros((2, 3), dtype=np.int32)
    mat_max = np.zeros((2, 3), dtype=np.int32)
    # Comparator bool array is per-column (length = matrix width).
    lower_better = np.zeros(3, dtype=np.bool_)
    # Exact (range-aware) cull — both search_inv branches.
    for search_inv in (True, False):
        ingf.compare_stat_vectors(
            mat_min[0], mat_max[0], mat_min[1], mat_max[1],
            lower_better, -1, search_inv,
        )
        ingf.pareto_filter_ingredients(mat_min, mat_max, lower_better, -1, search_inv)
    # Legacy fast cull (used when query.fast_cull=True).
    ingf.compare_stat_vectors_fast(mat_min[0], mat_min[1], lower_better, -1)
    ingf.pareto_filter_ingredients_fast(mat_min, lower_better, -1)


def _warm_search_engine():
    """
    Force the JIT compile of `dfs` and `search_meta_batch` (parallel=True is
    expensive to compile, 1-2s per process). M=1, k=1, S=36, N=1 — S=36 so the
    largest spell formula (any spell-mode spell, 36 deps in canonical layout
    after adding the SP-req slots) can read its slots. We invoke each kernel
    once per formula tag (fixed + every spell) so all branches compile eagerly.
    """
    M, k, S, N = 1, 1, 36, 1
    ings = np.zeros((M, 6), dtype=np.int32)
    void_eff = np.full((M, k), 100, dtype=np.int32)
    base_min = np.zeros((M, S), dtype=np.int32)
    base_max = np.zeros((M, S), dtype=np.int32)
    db_stat_min = np.zeros((N, S), dtype=np.int16)
    db_stat_max = np.zeros((N, S), dtype=np.int16)
    db_contrib_pos_mask = np.zeros((N, S), dtype=np.bool_)
    db_contrib_neg_mask = np.zeros((N, S), dtype=np.bool_)
    has_min_mask = np.zeros(S, dtype=np.bool_)
    has_max_mask = np.zeros(S, dtype=np.bool_)
    pos_weight_mask = np.zeros(S, dtype=np.bool_)
    neg_weight_mask = np.zeros(S, dtype=np.bool_)
    min_vals = np.zeros(S, dtype=np.int32)
    max_vals = np.zeros(S, dtype=np.int32)
    weights = np.zeros(S, dtype=np.float32)
    total_searched = np.zeros(1, dtype=np.int64)

    # One composite per warmup pass; positive weight ensures w>0 UB branches
    # compile. Flat dep_indices spans S so any spell formula (max 9 deps) and
    # shorter formulas all read valid slots without OOB.
    comp_count = 1
    comp_dep_indices = np.arange(S, dtype=np.int32)
    comp_dep_offset = np.array([0], dtype=np.int32)
    comp_dep_count = np.array([S], dtype=np.int32)
    comp_min = np.zeros(1, dtype=np.int32)
    comp_max = np.zeros(1, dtype=np.int32)
    comp_has_min = np.zeros(1, dtype=np.bool_)
    comp_has_max = np.zeros(1, dtype=np.bool_)
    comp_weight = np.array([1.0], dtype=np.float32)

    # Build context: skp=0, atk_speed=NORMAL (2.05), crit=0
    build_ctx = np.zeros(BUILD_CTX_SIZE, dtype=np.float64)
    build_ctx[5] = 2.05  # BUILD_CTX_ATK_SPD

    # Per-stat round-half-up bias for eff scaling (real Query passes 50 on req
    # stats); zero here is fine for warmup since we only need the kernel to
    # compile, not to compute meaningful values.
    round_offset = np.zeros(S, dtype=np.int32)

    # SP-cap-aware scoring metadata. Real Query marks SP stats active and
    # passes their ctx + corresponding xReq idx; warm-up uses all-inactive
    # to exercise the non-SP fast path. The req-idx array still needs to
    # type as int32, hence the explicit dtype.
    sp_score_active = np.zeros(S, dtype=np.bool_)
    sp_score_ctx = np.zeros(S, dtype=np.int32)
    sp_score_req_idx = np.full(S, -1, dtype=np.int32)
    sp_score_req_base = np.zeros(S, dtype=np.int32)

    void_eff_k2 = np.full((M, 2), 100, dtype=np.int32)

    # Cover every formula tag (fixed + every spell_id). Each loop iteration
    # warms all 3 kernels with that tag. After the first iteration the kernel
    # is fully compiled — subsequent iterations re-run the dispatch with the
    # new tag value to ensure the runtime branch for each spell is exercised.
    fixed_tags = (FORMULA_MUL_DIV_100, FORMULA_RAW_TO_PCT, FORMULA_EHP, FORMULA_EHPR)
    spell_tags = tuple(FORMULA_SPELL_DAMAGE_BASE + i for i in range(SPELL_COUNT))
    for tag in fixed_tags + spell_tags:
        comp_formula = np.array([tag], dtype=np.int32)

        # v2 ((m, i_0)-parallel) is the production path for k>=3. Compile it
        # eagerly here so the first real batch doesn't pay the parallel-jit cost.
        # (The legacy v1 `search_meta_batch` is no longer warmed/used — production
        # routes k=1/k=2 to the specialized kernels and k>=3 to v2.)
        se.search_meta_batch_v2(
            ings, k, void_eff, base_min, base_max,
            db_stat_min, db_stat_max, db_contrib_pos_mask, db_contrib_neg_mask,
            N, N, 0,
            has_min_mask, has_max_mask, pos_weight_mask, neg_weight_mask,
            min_vals, max_vals, weights, total_searched,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
            round_offset,
            sp_score_active, sp_score_ctx, sp_score_req_idx, sp_score_req_base,
        )


        se._search_meta_batch_k1(
            void_eff, base_min, base_max,
            db_stat_min, db_stat_max, N, N, 0,
            has_min_mask, has_max_mask, min_vals, max_vals, weights,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
            round_offset,
            sp_score_active, sp_score_ctx, sp_score_req_idx, sp_score_req_base,
        )

        se._search_meta_batch_k2(
            void_eff_k2, base_min, base_max,
            db_stat_min, db_stat_max, N, N, 0,
            has_min_mask, has_max_mask, min_vals, max_vals, weights,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
            round_offset,
            sp_score_active, sp_score_ctx, sp_score_req_idx, sp_score_req_base,
        )
