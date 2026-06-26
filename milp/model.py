"""
milp/model.py

Build the CP-SAT model for one crafting query. Full inversion, dual-bound.
See milp/README.md §3-4 and the plan.

Tight factorization (key to solver speed): the slot-p contribution to stat k is
    Σ_i s[i,k]·E[p]·y[p,i]  =  E[p] · (Σ_i s[i,k]·y[p,i])  =  E[p] · sv[p,k]
so there is ONE product per (slot, stat) — `E[p]·sv` — instead of one per
ingredient. `sv[p,k] = Σ_i s[i,k]·y[p,i]` is a tight linear "stat of the chosen
ingredient at slot p".

Inversion (negative E[p] swaps the roll bounds, mirroring search_engine.py:1228-
1233) is handled WITHOUT a sign binary: the build's max contribution at a slot is
    prod_hi = max(E·sv_hi, E·sv_lo)
and the min is `min(...)`, because sv_hi/sv_lo are the high/low rolls and the
larger of the two endpoint-products is the max whatever the sign of E.

Durability/duration is RAW (no effectiveness): its accumulator uses sv directly.

The model only reads these fields, so a SimpleNamespace works for unit tests:
  query : stat_count, weights_proj, min_proj, max_proj,
          has_min_mask_proj, has_max_mask_proj, dura_proj_idx
  recipe: base_min_stats_proj, base_max_stats_proj
  db    : stat_min_matrix [N,K], stat_max_matrix [N,K], json_ids [N], count
"""

import numpy as np
from ortools.sat.python import cp_model

from milp.objective import add_objective


class MilpModel:
    """Container for the CP-SAT model and its decision variables."""
    __slots__ = (
        "model", "query", "recipe", "db", "b", "N", "K", "dura_idx", "big",
        "y", "E", "A_min", "A_max", "dura_min", "dura_max",
        "E_min", "E_max", "weight_scale", "int_weights",
    )


def _prod_bounds(elo, ehi, slo, shi):
    c = (elo * slo, elo * shi, ehi * slo, ehi * shi)
    return min(c), max(c)


def _accum_bound(db, recipe, E_min, E_max):
    if db.count > 0:
        smax_abs = int(max(np.abs(db.stat_min_matrix).max(initial=0),
                           np.abs(db.stat_max_matrix).max(initial=0)))
    else:
        smax_abs = 0
    maxE = int(max(np.abs(E_min).max(initial=0), np.abs(E_max).max(initial=0)))
    base_abs = int(max(np.abs(recipe.base_min_stats_proj).max(initial=0),
                       np.abs(recipe.base_max_stats_proj).max(initial=0)))
    return 100 * base_abs + 6 * smax_abs * maxE + 1000


def build_model(db, b, E_min, E_max, query, recipe):
    """
    Returns a MilpModel with the objective set. `b` is [6,N,6] from
    milp/efficiency.build_b_tensor; E_min/E_max the per-slot eff bounds. Row order
    MUST match db.json_ids (post-IngredientDB-sort).
    """
    model = cp_model.CpModel()
    N = int(db.count)
    K = int(query.stat_count)
    dura_idx = int(query.dura_proj_idx)
    E_min = np.asarray(E_min, dtype=np.int64)
    E_max = np.asarray(E_max, dtype=np.int64)
    BIG = _accum_bound(db, recipe, E_min, E_max)

    smin = db.stat_min_matrix   # [N, K]
    smax = db.stat_max_matrix   # [N, K]
    base_lo = recipe.base_min_stats_proj
    base_hi = recipe.base_max_stats_proj

    # Per-stat column extrema → tight bounds for the sv "chosen-stat" variables.
    if N > 0:
        svhi_lo = smax.min(axis=0).astype(np.int64)
        svhi_hi = smax.max(axis=0).astype(np.int64)
        svlo_lo = smin.min(axis=0).astype(np.int64)
        svlo_hi = smin.max(axis=0).astype(np.int64)
    else:
        svhi_lo = svhi_hi = svlo_lo = svlo_hi = np.zeros(K, np.int64)

    # ---- variables ----
    y = [[model.NewBoolVar(f"y_{p}_{i}") for i in range(N)] for p in range(6)]
    E = [model.NewIntVar(int(E_min[p]), int(E_max[p]), f"E_{p}") for p in range(6)]

    A_min, A_max = {}, {}
    dura_min = dura_max = None

    # ---- (1) slot assignment ----
    for p in range(6):
        model.Add(sum(y[p]) == 1)

    # ---- (2) efficiency equation (sparse: only nonzero b terms) ----
    for p in range(6):
        terms = []
        for q in range(6):
            if q == p:
                continue
            col = b[q, :, p]
            nz = np.nonzero(col)[0]
            for i in nz:
                terms.append(int(col[i]) * y[q][int(i)])
        model.Add(E[p] == 100 + sum(terms))

    # ---- per-stat accumulators ----
    for k in range(K):
        kk = int(k)
        if kk == dura_idx:
            # durability: RAW, no eff. Sum the chosen stat over all slots.
            dura_max = model.NewIntVar(-BIG, BIG, "dura_max")
            dura_min = model.NewIntVar(-BIG, BIG, "dura_min")
            svh_slots, svl_slots = [], []
            for p in range(6):
                svh = model.NewIntVar(int(svhi_lo[kk]), int(svhi_hi[kk]), f"svh_{p}_{kk}")
                svl = model.NewIntVar(int(svlo_lo[kk]), int(svlo_hi[kk]), f"svl_{p}_{kk}")
                model.Add(svh == sum(int(smax[i, kk]) * y[p][i] for i in range(N)))
                model.Add(svl == sum(int(smin[i, kk]) * y[p][i] for i in range(N)))
                svh_slots.append(svh)
                svl_slots.append(svl)
            model.Add(dura_max == int(base_hi[kk]) + sum(svh_slots))
            model.Add(dura_min == int(base_lo[kk]) + sum(svl_slots))
            continue

        hi_terms, lo_terms = [], []
        for p in range(6):
            elo, ehi = int(E_min[p]), int(E_max[p])
            svh = model.NewIntVar(int(svhi_lo[kk]), int(svhi_hi[kk]), f"svh_{p}_{kk}")
            svl = model.NewIntVar(int(svlo_lo[kk]), int(svlo_hi[kk]), f"svl_{p}_{kk}")
            model.Add(svh == sum(int(smax[i, kk]) * y[p][i] for i in range(N)))
            model.Add(svl == sum(int(smin[i, kk]) * y[p][i] for i in range(N)))

            phlo, phhi = _prod_bounds(elo, ehi, int(svhi_lo[kk]), int(svhi_hi[kk]))
            pllo, plhi = _prod_bounds(elo, ehi, int(svlo_lo[kk]), int(svlo_hi[kk]))
            ph = model.NewIntVar(phlo, phhi, f"ph_{p}_{kk}")
            pl = model.NewIntVar(pllo, plhi, f"pl_{p}_{kk}")
            model.AddMultiplicationEquality(ph, [E[p], svh])
            model.AddMultiplicationEquality(pl, [E[p], svl])

            lo_b, hi_b = min(phlo, pllo), max(phhi, plhi)
            prod_hi = model.NewIntVar(lo_b, hi_b, f"phi_{p}_{kk}")
            prod_lo = model.NewIntVar(lo_b, hi_b, f"plo_{p}_{kk}")
            model.AddMaxEquality(prod_hi, [ph, pl])   # max contribution (any sign of E)
            model.AddMinEquality(prod_lo, [ph, pl])   # min contribution
            hi_terms.append(prod_hi)
            lo_terms.append(prod_lo)

        amax = model.NewIntVar(-BIG, BIG, f"Amax_{kk}")
        amin = model.NewIntVar(-BIG, BIG, f"Amin_{kk}")
        model.Add(amax == 100 * int(base_hi[kk]) + sum(hi_terms))
        model.Add(amin == 100 * int(base_lo[kk]) + sum(lo_terms))
        A_max[kk] = amax
        A_min[kk] = amin

    # ---- hard query constraints (both bounds, matching DFS leaf) ----
    has_min = query.has_min_mask_proj
    has_max = query.has_max_mask_proj
    Lp = query.min_proj
    Up = query.max_proj
    for k in range(K):
        kk = int(k)
        if kk == dura_idx:
            if has_min[kk]:
                model.Add(dura_max >= int(Lp[kk]))
            if has_max[kk]:
                model.Add(dura_min <= int(Up[kk]))
        else:
            if has_min[kk]:
                model.Add(A_max[kk] >= 100 * int(Lp[kk]))
            if has_max[kk]:
                model.Add(A_min[kk] <= 100 * int(Up[kk]))

    mm = MilpModel()
    mm.model = model
    mm.query = query
    mm.recipe = recipe
    mm.db = db
    mm.b = b
    mm.N = N
    mm.K = K
    mm.dura_idx = dura_idx
    mm.big = BIG
    mm.y = y
    mm.E = E
    mm.A_min = A_min
    mm.A_max = A_max
    mm.dura_min = dura_min
    mm.dura_max = dura_max
    mm.E_min = E_min
    mm.E_max = E_max

    add_objective(mm)
    return mm
