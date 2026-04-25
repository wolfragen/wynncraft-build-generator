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
    req_idx = np.full(5, -1, dtype=np.int32)

    msl.compare_vectors(mat[0], mat[1], 1, req_idx, -1)
    msl.pareto_filter(mat, 1, req_idx, -1)

    is_kept = np.ones(2, dtype=np.bool_)
    msl._intra_block_cull(0, 2, mat, 1, req_idx, -1, is_kept)

    survivors = np.array([0], dtype=np.int64)
    is_kept = np.ones(2, dtype=np.bool_)
    msl._parallel_check_block(1, 2, survivors, 1, mat, 1, req_idx, -1, is_kept)

    base_min = np.zeros((2, 1), dtype=np.int32)
    base_max = np.zeros((2, 1), dtype=np.int32)
    void_eff = np.zeros((2, 1), dtype=np.int32)
    msl._build_cull_matrix(base_min, base_max, void_eff, 1, req_idx)

    msl.pareto_filter_block(mat, 1, req_idx, -1, block_size=1)


def _warm_ingredient_filter():
    mat = np.zeros((2, 3), dtype=np.int32)
    req_idx = np.full(5, -1, dtype=np.int32)
    ingf.compare_stat_vectors(mat[0], mat[1], req_idx, -1)
    ingf.pareto_filter_ingredients(mat, req_idx, -1)


def _warm_search_engine():
    """
    Force the JIT compile of `dfs` and `search_meta_batch` (parallel=True is
    expensive to compile, 1-2s per process). M=1, k=1, S=8, N=1 — S=8 so the
    8-dep spell formula (Meteor) can read its slots. We run each kernel once
    per formula tag so every branch compiles eagerly.
    """
    M, k, S, N = 1, 1, 8, 1
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
    # compile. Flat dep_indices contains 8 entries [0..7] so the spell formula
    # (8 deps) and shorter formulas all read valid slots without OOB.
    comp_count = 1
    comp_dep_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    comp_dep_offset = np.array([0], dtype=np.int32)
    comp_dep_count = np.array([8], dtype=np.int32)
    comp_min = np.zeros(1, dtype=np.int32)
    comp_max = np.zeros(1, dtype=np.int32)
    comp_has_min = np.zeros(1, dtype=np.bool_)
    comp_has_max = np.zeros(1, dtype=np.bool_)
    comp_weight = np.array([1.0], dtype=np.float32)

    # Build context: skp=0, atk_speed=NORMAL (2.05), crit=0
    build_ctx = np.zeros(BUILD_CTX_SIZE, dtype=np.float64)
    build_ctx[5] = 2.05  # BUILD_CTX_ATK_SPD

    void_eff_k2 = np.full((M, 2), 100, dtype=np.int32)

    for tag in (FORMULA_MUL_DIV_100, FORMULA_RAW_TO_PCT, FORMULA_EHP,
                FORMULA_EHPR, FORMULA_SPELL_DAMAGE_BASE):  # last → spell_id=0 (Meteor)
        comp_formula = np.array([tag], dtype=np.int32)

        se.search_meta_batch(
            ings, k, void_eff, base_min, base_max,
            db_stat_min, db_stat_max, db_contrib_pos_mask, db_contrib_neg_mask,
            N, 0,
            has_min_mask, has_max_mask, pos_weight_mask, neg_weight_mask,
            min_vals, max_vals, weights, total_searched,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
        )

        se._search_meta_batch_k1(
            void_eff, base_min, base_max,
            db_stat_min, db_stat_max, N, 0,
            has_min_mask, has_max_mask, min_vals, max_vals, weights,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
        )

        se._search_meta_batch_k2(
            void_eff_k2, base_min, base_max,
            db_stat_min, db_stat_max, N, 0,
            has_min_mask, has_max_mask, min_vals, max_vals, weights,
            -1e18,
            comp_count, comp_formula,
            comp_dep_offset, comp_dep_count, comp_dep_indices,
            comp_min, comp_max, comp_has_min, comp_has_max, comp_weight,
            build_ctx,
        )
