"""
separable_search.py — exact crafting optimiser via objective separability.

A second, independent solver alongside the branch-and-bound DFS in
core/search_engine.py (which is left completely untouched). It is EXACT and
typically ~30x faster than the DFS on linear (composite-free) queries.

## Why it is fast

For a fixed effectiveness vector the leaf score is a weighted sum that DECOMPOSES
per slot, so the optimal void fill is per-slot argmax (O(k·N)), not the DFS's
O(N^k) branch-and-bound. Per META row:

    separable_opt(row) = score(base_min/max) + Σ_void best_score_marginal[void_eff]

is a VALID, TIGHT upper bound on that row's true (constrained) optimum. We process
all ~6M meta rows in descending bound order and branch-and-bound: every row whose
bound ≤ the best feasible build found is pruned. Only the few surviving rows get an
exact per-row constrained void-fill (a small DFS over that row's void slots with
score-suffix pruning + leaf feasibility on every min/max constraint). The per-row
*residual* budgeting (each fixed-meta part consumes part of the e.g. xReq≤45 budget)
is what a single global Lagrangian gets wrong.

## Scope (exact within this, else use the DFS)

- Composite stats (spell/EHP/EHPR/HPR) break separability -> rejected here
  (`comp_count > 0`); use main.py / the DFS for those.
- Skill-point cap scoring is treated linearly. Exact when no `_context` skill-point
  base is set (the cap 150-req never binds); for high base-SP `_context` it is an
  approximation — validate against the DFS.

Bit-exactness: the per-slot marginal uses the same `(stat*eff + round_offset)//100`
floor, 0.99/0.01 blend, raw durability and eff<0 swap as the DFS leaf.
"""

from time import time
import numpy as np
from numba import njit

from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from data.meta_set_loader import load_meta_sets


class UnsupportedQueryError(Exception):
    """Raised for queries outside the separable solver's exact scope (composites)."""


@njit(cache=True)
def _solve_row(void_eidx, base_score, base_cval, score_sorted, cmarg_sorted,
               best_score_per_eff, thr, sign, best_global):
    """
    Exact constrained void-fill for ONE meta row, via an explicit-stack DFS over
    that row's void slots.

      void_eidx        (k,)      eff-index per void slot
      base_score       scalar    recipe + fixed-meta score of the row
      base_cval        (C,)      recipe + fixed-meta value of each constrained stat
      score_sorted     (E,N)     per-eff blended score, DESC-sorted by score
      cmarg_sorted     (C,E,N)   per-eff constrained-stat roll, same sorted order
      best_score_per_eff (E,)    max score per eff (suffix bound)
      thr,sign         (C,)      threshold and +1(min-constraint)/-1(max-constraint)
      best_global      scalar    incumbent to beat (prunes)

    Returns (best_feasible_score, rank_choice[k]); score is -inf if no feasible fill
    beats best_global.
    """
    k = void_eidx.shape[0]
    C = thr.shape[0]
    N = score_sorted.shape[1]
    suf = np.zeros(k + 1)
    for j in range(k - 1, -1, -1):
        suf[j] = suf[j + 1] + best_score_per_eff[void_eidx[j]]

    best_loc = best_global
    choice = np.full(k, -1, np.int64)
    cur_choice = np.full(k, -1, np.int64)
    stack_r = np.zeros(k, np.int64)
    cum_score = np.zeros(k + 1)
    cum_cval = np.zeros((k + 1, C))
    cum_cval[0] = base_cval
    cum_score[0] = base_score
    j = 0
    if k > 0:
        stack_r[0] = 0
    while j >= 0:
        if j == k:
            ok = True
            for c in range(C):
                if sign[c] > 0:
                    if cum_cval[k, c] < thr[c]:
                        ok = False; break
                else:
                    if cum_cval[k, c] > thr[c]:
                        ok = False; break
            if ok and cum_score[k] > best_loc:
                best_loc = cum_score[k]
                for jj in range(k):
                    choice[jj] = cur_choice[jj]
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        ei = void_eidx[j]
        r = stack_r[j]
        if r >= N or cum_score[j] + score_sorted[ei, r] + suf[j + 1] <= best_loc:
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        cur_choice[j] = r
        cum_score[j + 1] = cum_score[j] + score_sorted[ei, r]
        for c in range(C):
            cum_cval[j + 1, c] = cum_cval[j, c] + cmarg_sorted[c, ei, r]
        j += 1
        if j < k:
            stack_r[j] = 0
    return best_loc, choice


def solve_separable(user_query, skill, item_type, tier, lvl_min, lvl_max,
                    search_for_inversion=True, base_path="data/precalc/generic_cull",
                    consumable=False, ingredients_raw=None, recipes=None, verbose=False):
    """
    Find the globally optimal crafted build for a linear (composite-free) query.

    Returns a dict:
      build        list[int]  6 ingredient JSON ids in slot order 0..5
      score        float      the leaf score of that build (this solver's objective)
      meta_n       int        number of fixed meta ingredients in the winning row
      status       str        "OPTIMAL" or "INFEASIBLE"
      rows_solved  int        rows that needed the constrained solve (of total)
      timings      dict       load / prep / search seconds

    Raises UnsupportedQueryError for composite queries.
    """
    q = build_query(user_json=user_query, search_for_inversion=search_for_inversion,
                    item_type=item_type, skill=skill, consumable=consumable)
    if q.comp_count > 0:
        raise UnsupportedQueryError(
            f"Composite stats are outside the separable solver's exact scope "
            f"(query defines {q.comp_count}). Use the DFS (main.py).")

    if ingredients_raw is None:
        ingredients_raw = load_ingredients("data/ingreds_compress.json")
    if recipes is None:
        recipes = load_recipes("data/recipes_compress.json")
    recipe_raw = find_recipe(recipes=recipes, item_type=item_type, skill=skill,
                             lvl_min=lvl_min, lvl_max=lvl_max)
    rec = build_recipe(recipe_raw, q, tier=tier)

    normals = filter_raw_ingredients(ingredients_raw, q, rec, cull=False)
    db = IngredientDB(normals, q)
    smin = db.stat_min_matrix.astype(np.int64); smax = db.stat_max_matrix.astype(np.int64)
    K = q.stat_count; N = smin.shape[0]
    w = q.weights_proj.astype(np.float64); roi = q.round_offset_proj.astype(np.int64)
    dura = int(q.dura_proj_idx)

    cons = []
    for s in range(K):
        if q.has_min_mask_proj[s]:
            cons.append((s, +1, int(q.min_proj[s])))
        if q.has_max_mask_proj[s]:
            cons.append((s, -1, int(q.max_proj[s])))
    C = len(cons)
    sign = np.array([c[1] for c in cons], np.int64)
    thr = np.array([c[2] for c in cons], np.float64)

    t_load = time()
    batches = load_meta_sets(skill, q, rec, culling=True, max_cull=int(q.suggested_max_cull),
                             base_path=base_path)
    load_s = time() - t_load

    t = time()
    eff_lists = [b.void_eff_matrix.ravel() for b in batches if b.void_eff_matrix.size]
    all_effs = np.unique(np.concatenate(eff_lists + [np.array([100], np.int32)])).astype(np.int32)
    E = len(all_effs)

    score_eff = np.zeros((E, N))
    cmarg = np.zeros((C, E, N))
    for ei, eff in enumerate(all_effs):
        eff = int(eff)
        if eff >= 0:
            cmax = (smax * eff + roi) // 100; cmin = (smin * eff + roi) // 100
        else:
            cmax = (smin * eff + roi) // 100; cmin = (smax * eff + roi) // 100
        if dura >= 0:
            cmax[:, dura] = smax[:, dura]; cmin[:, dura] = smin[:, dura]
        score_eff[ei] = (cmax * 0.99 + cmin * 0.01) @ w
        for ci, (s, sg, _) in enumerate(cons):
            cmarg[ci, ei] = cmax[:, s] if sg > 0 else cmin[:, s]

    order = np.argsort(-score_eff, axis=1)
    score_sorted = np.ascontiguousarray(np.take_along_axis(score_eff, order, axis=1))
    cmarg_sorted = np.zeros((C, E, N))
    for ci in range(C):
        cmarg_sorted[ci] = np.take_along_axis(cmarg[ci], order, axis=1)
    cmarg_sorted = np.ascontiguousarray(cmarg_sorted)
    best_score_per_eff = np.ascontiguousarray(score_sorted[:, 0].copy())

    cstat_idx = np.array([c[0] for c in cons], np.int64)
    cstat_max = sign > 0    # True -> use base_max column, else base_min
    pre = []
    all_bounds = []; all_n = []; all_m = []
    for n, b in enumerate(batches):
        M = b.ings_matrix.shape[0]
        if M == 0:
            pre.append(None); continue
        imax = b.base_max_matrix; imin = b.base_min_matrix          # int32, no float copy
        # matmul-then-blend: reduce (M,K) to (M,) BEFORE the float blend (avoids the
        # (M,K) float64 intermediate that dominated prep).
        base_score = (imax @ w) * 0.99 + (imin @ w) * 0.01
        # NOTE: the per-row constrained-stat base (base_cval) is computed LAZILY in the
        # solve loop below — only ~thousands of rows survive bound-pruning and get
        # solved, so materialising it for all ~6M rows here was pure waste (~0.9s).
        veidx = np.zeros((M, b.void_count), np.int64)
        bound = base_score.copy()
        for j in range(b.void_count):
            ix = np.searchsorted(all_effs, b.void_eff_matrix[:, j])
            veidx[:, j] = ix
            bound += best_score_per_eff[ix]
        pre.append((M, base_score, imax, imin, veidx))
        all_bounds.append(bound)
        all_n.append(np.full(M, n, np.int64))
        all_m.append(np.arange(M, dtype=np.int64))
    bounds_all = np.concatenate(all_bounds)
    n_all = np.concatenate(all_n); m_all = np.concatenate(all_m)
    order_rows = np.argsort(-bounds_all)
    prep_s = time() - t

    t2 = time()
    best = -1e18; best_rc = None; best_n = best_m = -1; solved = 0
    for oi in order_rows:
        if bounds_all[oi] <= best:
            break
        n = int(n_all[oi]); m = int(m_all[oi])
        _, base_score, imax, imin, veidx = pre[n]
        # constrained-stat base for THIS row only (min-constraint uses max-roll base,
        # max-constraint uses min-roll base).
        base_cval = np.where(cstat_max, imax[m, cstat_idx], imin[m, cstat_idx]).astype(np.float64)
        sc, choice = _solve_row(veidx[m], float(base_score[m]), base_cval,
                                score_sorted, cmarg_sorted, best_score_per_eff,
                                thr, sign, best)
        solved += 1
        if sc > best:
            best = sc; best_rc = choice.copy(); best_n = n; best_m = m
    search_s = time() - t2

    if best_n < 0:
        return {"build": None, "score": None, "status": "INFEASIBLE",
                "meta_n": -1, "rows_solved": solved,
                "timings": {"load": load_s, "prep": prep_s, "search": search_s}}

    b = batches[best_n]
    build = [int(x) for x in b.ings_matrix[best_m]]
    for j in range(b.void_count):
        slot = int(b.void_slots_matrix[best_m, j]); eff = int(b.void_eff_matrix[best_m, j])
        ei = int(np.searchsorted(all_effs, eff))
        build[slot] = int(db.json_ids[order[ei, best_rc[j]]])

    if verbose:
        print(f"[separable] load={load_s:.1f}s prep={prep_s:.2f}s search={search_s:.2f}s "
              f"rows_solved={solved}/{bounds_all.shape[0]} score={best:.1f}")
    return {"build": build, "score": best, "status": "OPTIMAL", "meta_n": best_n,
            "rows_solved": solved, "recipe_raw": recipe_raw,
            "timings": {"load": load_s, "prep": prep_s, "search": search_s}}
