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

- Composite stats break per-slot separability, but a SINGLE monotone composite is
  handled exactly via a composite-aware per-row solve (track the composite's dep roll
  totals, evaluate it at the leaf, prune with a node-local upper bound that tightens
  as deps are fixed): HPR = rawToPct, EHP/EHPR = numerator / D(def,agi), and spell /
  melee average damage (reusing the engine's corner evaluators; monotone non-
  decreasing in every dep). Multiple composites still fall back to the DFS.
- Skill-point cap scoring is treated linearly for the LINEAR objective part. Exact
  when no `_context` skill-point base is set; for high base-SP `_context` it is an
  approximation — validate against the DFS.

Bit-exactness: the per-slot marginal uses the same `(stat*eff + round_offset)//100`
floor, 0.99/0.01 blend, raw durability and eff<0 swap as the DFS leaf; the composite
leaf/bound formulas mirror core/search_engine exactly (the spell path calls the very
same njit corner functions, so it shares the DFS's monotonicity assumption).
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
from data.skillpoint_lookup import SKP_HEADLINE_PCT, SKP_DEF, SKP_AGI, SKP_MAX
from data.spells import SPELL_USE_SPELL, get_spell_deps
from core.search_engine import (_eval_spell_corner_spell, _eval_spell_corner_melee,
                                _BCTX_WD_N)

# Skill-point -> headline % rows for the EHP/EHPR denominator (mirror search_engine).
_SKP_DEF_ROW = np.ascontiguousarray(SKP_HEADLINE_PCT[SKP_DEF])   # (151,) float64
_SKP_AGI_ROW = np.ascontiguousarray(SKP_HEADLINE_PCT[SKP_AGI])
_SKP_MAX = int(SKP_MAX)

# Composite type codes.
_C_HPR = 0    # rawToPct(hprRaw, hprPct)
_C_EHP = 1    # max(5,hpBonus) / D(def,agi)
_C_EHPR = 2   # rawToPct(hprRaw,hprPct) / D(def,agi)
_C_SPELL = 3  # average spell/melee damage (monotone in every dep) — variable arity


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


# ---------------------------------------------------------------------------
# Composite support: HPR / EHP / EHPR (the monotone, non-spell composites).
# All leaf/bound formulas mirror core/search_engine bit-for-bit.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _rtp(r, d):
    """rawToPct, matching core/search_engine._raw_to_pct and the oracle."""
    if r > 0:
        return (r * (100 + d)) // 100
    if r < 0:
        v = (r * (100 - d)) // 100
        return v if v < 0 else 0
    return 0


@njit(cache=True)
def _clamp_skp(c):
    if c < 0:
        return np.int64(0)
    if c > _SKP_MAX:
        return np.int64(_SKP_MAX)
    return np.int64(c)


@njit(cache=True)
def _denom(def_lo, def_hi, agi_lo, agi_hi):
    """Range of D = 1 - (1-agi_pct)*def_pct over the def/agi count ranges."""
    dl = _clamp_skp(def_lo); dh = _clamp_skp(def_hi)
    al = _clamp_skp(agi_lo); ah = _clamp_skp(agi_hi)
    d_min = 1.0 - (1.0 - _SKP_AGI_ROW[al]) * _SKP_DEF_ROW[dh]
    d_max = 1.0 - (1.0 - _SKP_AGI_ROW[ah]) * _SKP_DEF_ROW[dl]
    return d_min, d_max


@njit(cache=True)
def _ehp_bounds(hp_lo, hp_hi, def_lo, def_hi, agi_lo, agi_hi):
    if hp_lo < 5: hp_lo = 5
    if hp_hi < 5: hp_hi = 5
    d_min, d_max = _denom(def_lo, def_hi, agi_lo, agi_hi)
    return np.int64(hp_lo / d_max), np.int64(hp_hi / d_min)


@njit(cache=True)
def _rtp_bounds(raw_lo, raw_hi, dl, dh):
    """Admissible bounds on rawToPct over the box (mirror search_engine)."""
    first = True; lo = np.int64(0); hi = np.int64(0)
    if raw_hi > 0:
        rl = raw_lo if raw_lo > 0 else np.int64(0)
        rh = np.int64(raw_hi)
        c1 = rl * (100 + dl); c2 = rl * (100 + dh)
        c3 = rh * (100 + dl); c4 = rh * (100 + dh)
        p_lo = min(min(c1, c2), min(c3, c4)) // 100
        p_hi = max(max(c1, c2), max(c3, c4)) // 100
        lo = p_lo; hi = p_hi; first = False
    if raw_lo < 0:
        rl = np.int64(raw_lo); rh = raw_hi if raw_hi < 0 else np.int64(0)
        c1 = rl * (100 - dl); c2 = rl * (100 - dh)
        c3 = rh * (100 - dl); c4 = rh * (100 - dh)
        g_lo = min(min(c1, c2), min(c3, c4)) // 100
        g_hi = max(max(c1, c2), max(c3, c4)) // 100
        if g_lo > 0: g_lo = np.int64(0)
        if g_hi > 0: g_hi = np.int64(0)
        if first:
            lo = g_lo; hi = g_hi
        else:
            if g_lo < lo: lo = g_lo
            if g_hi > hi: hi = g_hi
    return lo, hi


@njit(cache=True)
def _ehpr_bounds(raw_lo, raw_hi, pct_lo, pct_hi, def_lo, def_hi, agi_lo, agi_hi):
    hpr_lo, hpr_hi = _rtp_bounds(raw_lo, raw_hi, pct_lo, pct_hi)
    d_min, d_max = _denom(def_lo, def_hi, agi_lo, agi_hi)
    c1 = hpr_lo / d_min; c2 = hpr_lo / d_max
    c3 = hpr_hi / d_min; c4 = hpr_hi / d_max
    return np.int64(min(min(c1, c2), min(c3, c4))), np.int64(max(max(c1, c2), max(c3, c4)))


# --- vectorised helpers for the per-row composite upper bound (over many rows) ---
def _rtp_np(r, d):
    """Vectorised rawToPct over int arrays (matches _rtp)."""
    r = r.astype(np.int64); d = d.astype(np.int64)
    out = np.zeros(r.shape, np.int64)
    pos = r > 0
    out[pos] = (r[pos] * (100 + d[pos])) // 100
    neg = r < 0
    if neg.any():
        out[neg] = np.minimum((r[neg] * (100 - d[neg])) // 100, 0)
    return out


def _denom_min_np(def_hi, agi_lo):
    """Smallest D over the box (max def, min agi) -> largest EHP/EHPR (for num>0)."""
    d = np.clip(def_hi, 0, _SKP_MAX).astype(np.int64)
    a = np.clip(agi_lo, 0, _SKP_MAX).astype(np.int64)
    return 1.0 - (1.0 - _SKP_AGI_ROW[a]) * _SKP_DEF_ROW[d]


def _denom_max_np(def_lo, agi_hi):
    """Largest D over the box (min def, max agi) -> for the num<0 EHPR corner."""
    d = np.clip(def_lo, 0, _SKP_MAX).astype(np.int64)
    a = np.clip(agi_hi, 0, _SKP_MAX).astype(np.int64)
    return 1.0 - (1.0 - _SKP_AGI_ROW[a]) * _SKP_DEF_ROW[d]


@njit(cache=True)
def _node_ub(comp_type, cdmax, cdmin, sufhi, suflo, j):
    """
    RAW composite upper bound for the subtree at node j: each remaining void slot is
    replaced by its per-eff favourable extreme (max marginal for the numerator/def,
    min marginal for agi), so the (monotone) composite over-estimates any completion.
    Returns the unweighted composite value (callers apply the weight for the objective
    bound, and compare it directly against a composite min for the FEASIBILITY prune —
    which must be weight-independent, since a `min` constraint can carry weight 0).
    cdmax/cdmin are cum_dmax[j]/cum_dmin[j]; sufhi/suflo are the favourable suffix sums.
    """
    hi0 = cdmax[0] + sufhi[0, j]; hi1 = cdmax[1] + sufhi[1, j]
    hi2 = cdmax[2] + sufhi[2, j]; hi3 = cdmax[3] + sufhi[3, j]
    lo2 = cdmin[2] + suflo[2, j]; lo3 = cdmin[3] + suflo[3, j]
    if comp_type == 0:        # HPR
        return float(_rtp(np.int64(hi0), np.int64(hi1)))
    if comp_type == 1:        # EHP: max(5,hp) / D(max def, min agi)
        dmn, _ = _denom(np.int64(hi1), np.int64(hi1), np.int64(lo2), np.int64(lo2))
        hh = hi0 if hi0 > 5.0 else 5.0
        return hh / dmn
    # EHPR: numerator / D, two-sided in D to cover the numerator's sign
    num = _rtp(np.int64(hi0), np.int64(hi1))
    dmn, _ = _denom(np.int64(hi2), np.int64(hi2), np.int64(lo3), np.int64(lo3))
    _, dmx = _denom(np.int64(lo2), np.int64(lo2), np.int64(hi3), np.int64(hi3))
    v1 = num / dmn; v2 = num / dmx
    return v1 if v1 > v2 else v2


@njit(cache=True)
def _solve_row_comp(void_eidx, base_score, base_cval, base_dmin, base_dmax,
                    score_sorted, cmarg_sorted, dmin_s, dmax_s, dmax_emax, dmin_emin,
                    n_deps, comp_type, best_score_per_eff, thr, sign, w_comp,
                    has_cmin, cmin_thr, has_cmax, cmax_thr, best_global):
    """
    Exact per-row constrained void-fill WITH a monotone composite (HPR/EHP/EHPR).
    Accumulates the composite's dep roll totals and adds w*(0.99*cmax + 0.01*cmin) at
    the leaf; prunes on the linear suffix bound plus a NODE-LOCAL composite upper
    bound (_node_ub) that tightens as deps are fixed down the void-slot DFS — without
    it, when the composite dominates a tiny linear part the per-row solve degenerates
    to brute force. dmax_emax/dmin_emin are per-eff favourable extremes (4,E). Dep
    order per type matches the engine leaf:
      HPR [hprRaw, hprPct];  EHP [hpBonus, def, agi];  EHPR [hprRaw, hprPct, def, agi].
    """
    k = void_eidx.shape[0]; C = thr.shape[0]; N = score_sorted.shape[1]
    suf = np.zeros(k + 1)
    sufhi = np.zeros((4, k + 1)); suflo = np.zeros((4, k + 1))
    for j in range(k - 1, -1, -1):
        ei = void_eidx[j]
        suf[j] = suf[j + 1] + best_score_per_eff[ei]
        for d in range(4):
            sufhi[d, j] = sufhi[d, j + 1] + dmax_emax[d, ei]
            suflo[d, j] = suflo[d, j + 1] + dmin_emin[d, ei]
    best_loc = best_global
    choice = np.full(k, -1, np.int64)
    cur_choice = np.full(k, -1, np.int64)
    stack_r = np.zeros(k, np.int64)
    cum_score = np.zeros(k + 1)
    cum_cval = np.zeros((k + 1, C))
    cum_dmin = np.zeros((k + 1, 4)); cum_dmax = np.zeros((k + 1, 4))
    comp_raw_node = np.zeros(k + 1)   # RAW (unweighted) composite UB per node
    cum_cval[0] = base_cval; cum_score[0] = base_score
    for dd in range(4):
        cum_dmin[0, dd] = base_dmin[dd]; cum_dmax[0, dd] = base_dmax[dd]
    j = 0
    if k > 0:
        stack_r[0] = 0
        comp_raw_node[0] = _node_ub(comp_type, cum_dmax[0], cum_dmin[0], sufhi, suflo, 0)
    while j >= 0:
        if j == k:
            if comp_type == 0:        # HPR
                ca = _rtp(np.int64(cum_dmin[k, 0]), np.int64(cum_dmin[k, 1]))
                cb = _rtp(np.int64(cum_dmax[k, 0]), np.int64(cum_dmax[k, 1]))
                cmn = ca if ca < cb else cb
                cmx = cb if ca < cb else ca
            elif comp_type == 1:      # EHP
                cmn, cmx = _ehp_bounds(np.int64(cum_dmin[k, 0]), np.int64(cum_dmax[k, 0]),
                                       np.int64(cum_dmin[k, 1]), np.int64(cum_dmax[k, 1]),
                                       np.int64(cum_dmin[k, 2]), np.int64(cum_dmax[k, 2]))
            else:                     # EHPR
                cmn, cmx = _ehpr_bounds(np.int64(cum_dmin[k, 0]), np.int64(cum_dmax[k, 0]),
                                        np.int64(cum_dmin[k, 1]), np.int64(cum_dmax[k, 1]),
                                        np.int64(cum_dmin[k, 2]), np.int64(cum_dmax[k, 2]),
                                        np.int64(cum_dmin[k, 3]), np.int64(cum_dmax[k, 3]))
            ok = True
            if has_cmin and cmx < cmin_thr:
                ok = False
            if has_cmax and cmn > cmax_thr:
                ok = False
            if ok:
                for c in range(C):
                    if sign[c] > 0:
                        if cum_cval[k, c] < thr[c]:
                            ok = False; break
                    else:
                        if cum_cval[k, c] > thr[c]:
                            ok = False; break
            if ok:
                total = cum_score[k] + w_comp * (0.99 * cmx + 0.01 * cmn)
                if total > best_loc:
                    best_loc = total
                    for jj in range(k):
                        choice[jj] = cur_choice[jj]
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        ei = void_eidx[j]; r = stack_r[j]
        # Prune the subtree if: (a) it is exhausted, (b) its objective bound can't beat
        # the incumbent, or (c) FEASIBILITY: even the max composite over the subtree
        # falls short of a required composite min (weight-independent, so it works with
        # no incumbent -> infeasible queries collapse fast instead of brute-forcing).
        if (r >= N
                or cum_score[j] + score_sorted[ei, r] + suf[j + 1] + w_comp * comp_raw_node[j] <= best_loc
                or (has_cmin and comp_raw_node[j] < cmin_thr)):
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        cur_choice[j] = r
        cum_score[j + 1] = cum_score[j] + score_sorted[ei, r]
        for c in range(C):
            cum_cval[j + 1, c] = cum_cval[j, c] + cmarg_sorted[c, ei, r]
        for dd in range(4):
            cum_dmin[j + 1, dd] = cum_dmin[j, dd]
            cum_dmax[j + 1, dd] = cum_dmax[j, dd]
        for dd in range(n_deps):
            cum_dmin[j + 1, dd] += dmin_s[dd, ei, r]
            cum_dmax[j + 1, dd] += dmax_s[dd, ei, r]
        j += 1
        if j < k:
            stack_r[j] = 0
            comp_raw_node[j] = _node_ub(comp_type, cum_dmax[j], cum_dmin[j], sufhi, suflo, j)
    return best_loc, choice


# ---------------------------------------------------------------------------
# Composite support: SPELL (average spell/melee damage).
#
# The engine's average-damage corner (core/search_engine._eval_spell_corner_*)
# is monotone non-decreasing in every dep (positive weapon/SP/raw contributions,
# non-negative mults & total_convert, SP clamps preserve monotonicity). So:
#   leaf  cmax = corner(deps_max_roll),  cmin = corner(deps_min_roll)
#   node-local UB = corner(cum_dmax[j] + favourable_max_suffix)  (max side only)
# We reuse the engine corner fns verbatim (numba can call njit from njit) — never
# re-derive the spell formula. ND = 36 (spell) or 32 (melee).
# ---------------------------------------------------------------------------
@njit(cache=True)
def _spell_corner(deps, use_spell, spell_id, ctx):
    """Dispatch one corner to the right engine evaluator (static branch for numba)."""
    if use_spell:
        return _eval_spell_corner_spell(deps, spell_id, ctx)
    return _eval_spell_corner_melee(deps, spell_id, ctx)


@njit(cache=True)
def _spell_bounds_all(d_hi, spell_id, use_spell, ctx):
    """
    Per-row favourable spell value: d_hi is (M, ND) (the favourable max-roll dep
    total per row); evaluate the (monotone) corner for each row -> (M,) float64.
    """
    M = d_hi.shape[0]
    ND = d_hi.shape[1]
    out = np.zeros(M)
    deps = np.empty(ND)
    for m in range(M):
        for d in range(ND):
            deps[d] = d_hi[m, d]
        out[m] = _spell_corner(deps, use_spell, spell_id, ctx)
    return out


@njit(cache=True)
def _solve_row_spell(void_eidx, base_score, base_cval, base_dmin, base_dmax,
                     score_sorted, cmarg_sorted, dmin_s, dmax_s, dmax_emax,
                     ND, spell_id, use_spell, ctx, best_score_per_eff, thr, sign,
                     w_comp, has_cmin, cmin_thr, has_cmax, cmax_thr, best_global):
    """
    Exact per-row constrained void-fill WITH the (monotone) spell composite.

    Same explicit-stack void-slot DFS as _solve_row_comp, but the dep totals are
    width ND (32/36) instead of 4, evaluated through the engine corner fns:
      leaf  cmx = corner(cum_dmax[k]),  cmn = corner(cum_dmin[k])  -> w*(0.99cmx+0.01cmn)
      node  comp_raw_node[j] = corner(cum_dmax[j] + sufhi[:, j])     (RAW, max side only)
    The raw node bound drives both the objective prune (x w) and the composite-min
    FEASIBILITY prune (raw vs cmin_thr), so an infeasible min collapses with no
    incumbent. sufhi (ND, k+1) are favourable-max suffix sums; no suflo (monotone).
    Dep order is canonical get_spell_deps(spell_id) (== comp["deps"]).
    """
    k = void_eidx.shape[0]; C = thr.shape[0]; N = score_sorted.shape[1]
    suf = np.zeros(k + 1)
    sufhi = np.zeros((ND, k + 1))
    for j in range(k - 1, -1, -1):
        ei = void_eidx[j]
        suf[j] = suf[j + 1] + best_score_per_eff[ei]
        for d in range(ND):
            sufhi[d, j] = sufhi[d, j + 1] + dmax_emax[d, ei]
    best_loc = best_global
    choice = np.full(k, -1, np.int64)
    cur_choice = np.full(k, -1, np.int64)
    stack_r = np.zeros(k, np.int64)
    cum_score = np.zeros(k + 1)
    cum_cval = np.zeros((k + 1, C))
    cum_dmin = np.zeros((k + 1, ND)); cum_dmax = np.zeros((k + 1, ND))
    comp_raw_node = np.zeros(k + 1)   # RAW (unweighted) spell-damage UB per node
    deps_hi = np.empty(ND); deps_lo = np.empty(ND); deps_nd = np.empty(ND)
    cum_cval[0] = base_cval; cum_score[0] = base_score
    for dd in range(ND):
        cum_dmin[0, dd] = base_dmin[dd]; cum_dmax[0, dd] = base_dmax[dd]
    j = 0
    if k > 0:
        stack_r[0] = 0
        for dd in range(ND):
            deps_nd[dd] = cum_dmax[0, dd] + sufhi[dd, 0]
        comp_raw_node[0] = _spell_corner(deps_nd, use_spell, spell_id, ctx)
    while j >= 0:
        if j == k:
            for dd in range(ND):
                deps_hi[dd] = cum_dmax[k, dd]; deps_lo[dd] = cum_dmin[k, dd]
            cmx = _spell_corner(deps_hi, use_spell, spell_id, ctx)
            cmn = _spell_corner(deps_lo, use_spell, spell_id, ctx)
            ok = True
            if has_cmin and cmx < cmin_thr:
                ok = False
            if has_cmax and cmn > cmax_thr:
                ok = False
            if ok:
                for c in range(C):
                    if sign[c] > 0:
                        if cum_cval[k, c] < thr[c]:
                            ok = False; break
                    else:
                        if cum_cval[k, c] > thr[c]:
                            ok = False; break
            if ok:
                total = cum_score[k] + w_comp * (0.99 * cmx + 0.01 * cmn)
                if total > best_loc:
                    best_loc = total
                    for jj in range(k):
                        choice[jj] = cur_choice[jj]
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        ei = void_eidx[j]; r = stack_r[j]
        # (a) exhausted, (b) objective bound can't beat incumbent, or (c) FEASIBILITY:
        # max spell damage over the subtree can't reach a required min (weight-free, so
        # it prunes infeasible queries with no incumbent instead of brute-forcing).
        if (r >= N
                or cum_score[j] + score_sorted[ei, r] + suf[j + 1] + w_comp * comp_raw_node[j] <= best_loc
                or (has_cmin and comp_raw_node[j] < cmin_thr)):
            j -= 1
            if j >= 0:
                stack_r[j] += 1
            continue
        cur_choice[j] = r
        cum_score[j + 1] = cum_score[j] + score_sorted[ei, r]
        for c in range(C):
            cum_cval[j + 1, c] = cum_cval[j, c] + cmarg_sorted[c, ei, r]
        for dd in range(ND):
            cum_dmin[j + 1, dd] = cum_dmin[j, dd] + dmin_s[dd, ei, r]
            cum_dmax[j + 1, dd] = cum_dmax[j, dd] + dmax_s[dd, ei, r]
        j += 1
        if j < k:
            stack_r[j] = 0
            for dd in range(ND):
                deps_nd[dd] = cum_dmax[j, dd] + sufhi[dd, j]
            comp_raw_node[j] = _spell_corner(deps_nd, use_spell, spell_id, ctx)
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
    from data.stats import (FORMULA_RAW_TO_PCT, FORMULA_EHP, FORMULA_EHPR,
                            FORMULA_SPELL_DAMAGE_BASE)
    q = build_query(user_json=user_query, search_for_inversion=search_for_inversion,
                    item_type=item_type, skill=skill, consumable=consumable)
    # Composite support: a single monotone composite (HPR / EHP / EHPR / SPELL) is
    # handled exactly via the meta-row B&B + composite-aware per-row solve. Dep order
    # mirrors the engine leaf. Anything else (multiple composites) falls back to the DFS.
    _CTYPE = {FORMULA_RAW_TO_PCT: (_C_HPR, 2), FORMULA_EHP: (_C_EHP, 3),
              FORMULA_EHPR: (_C_EHPR, 4)}
    comp = None
    if q.comp_count > 0:
        f0 = int(q.comp_formula[0])
        if q.comp_count == 1 and f0 in _CTYPE:
            ctype, ndeps = _CTYPE[f0]
            off = int(q.comp_dep_offset[0])
            comp = {"type": ctype, "n": ndeps,
                    "deps": [int(q.comp_dep_indices[off + i]) for i in range(ndeps)],
                    "w": float(q.comp_weight[0]),
                    "has_min": bool(q.comp_has_min[0]), "min": float(q.comp_min[0]),
                    "has_max": bool(q.comp_has_max[0]), "max": float(q.comp_max[0])}
        elif q.comp_count == 1 and f0 >= FORMULA_SPELL_DAMAGE_BASE:
            sid = f0 - FORMULA_SPELL_DAMAGE_BASE
            ndeps = int(q.comp_dep_count[0])
            off = int(q.comp_dep_offset[0])
            comp = {"type": _C_SPELL, "spell_id": sid,
                    "use_spell": bool(SPELL_USE_SPELL[sid]), "n": ndeps,
                    "deps": [int(q.comp_dep_indices[off + i]) for i in range(ndeps)],
                    "w": float(q.comp_weight[0]),
                    "has_min": bool(q.comp_has_min[0]), "min": float(q.comp_min[0]),
                    "has_max": bool(q.comp_has_max[0]), "max": float(q.comp_max[0])}
        else:
            raise UnsupportedQueryError(
                f"Only a single HPR/EHP/EHPR/SPELL composite is supported so far "
                f"(query defines {q.comp_count}, formula {f0}). Use the DFS (main.py).")

    if ingredients_raw is None:
        ingredients_raw = load_ingredients("data/ingreds_compress.json")
    if recipes is None:
        recipes = load_recipes("data/recipes_compress.json")
    recipe_raw = find_recipe(recipes=recipes, item_type=item_type, skill=skill,
                             lvl_min=lvl_min, lvl_max=lvl_max)
    rec = build_recipe(recipe_raw, q, tier=tier)

    # Spell composite: overlay the weapon recipe's neutral damage onto the build
    # ctx (mirrors core/search_engine.search_pipelined ~3506-3516) so the spell
    # kernel scales the right intrinsic weapon damage instead of seeing WD_N=0.
    is_spell = comp is not None and comp["type"] == _C_SPELL
    spell_ctx = None
    if is_spell:
        spell_ctx = np.array(q.build_ctx, np.float64).copy()
        if rec.weapon_dam_neutral and (rec.weapon_dam_neutral[0] or rec.weapon_dam_neutral[1]):
            spell_ctx[_BCTX_WD_N] = (rec.weapon_dam_neutral[0] + rec.weapon_dam_neutral[1]) / 2.0

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
    if verbose:
        print(f"[separable] loading + culling meta sets ({skill}, max_cull={int(q.suggested_max_cull)}):")
    batches = load_meta_sets(skill, q, rec, culling=True, max_cull=int(q.suggested_max_cull),
                             should_print=verbose, base_path=base_path)
    load_s = time() - t_load

    t = time()
    eff_lists = [b.void_eff_matrix.ravel() for b in batches if b.void_eff_matrix.size]
    all_effs = np.unique(np.concatenate(eff_lists + [np.array([100], np.int32)])).astype(np.int32)
    E = len(all_effs)

    score_eff = np.zeros((E, N))
    cmarg = np.zeros((C, E, N))
    # Composite: per-eff roll contributions (max/min) of each dep, in dep order.
    nd = comp["n"] if comp is not None else 0
    dmax_e = np.zeros((nd, E, N)) if comp is not None else None
    dmin_e = np.zeros((nd, E, N)) if comp is not None else None
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
        if comp is not None:
            for d in range(nd):
                sidx = comp["deps"][d]
                dmax_e[d, ei] = cmax[:, sidx]; dmin_e[d, ei] = cmin[:, sidx]

    order = np.argsort(-score_eff, axis=1)
    score_sorted = np.ascontiguousarray(np.take_along_axis(score_eff, order, axis=1))
    cmarg_sorted = np.zeros((C, E, N))
    for ci in range(C):
        cmarg_sorted[ci] = np.take_along_axis(cmarg[ci], order, axis=1)
    cmarg_sorted = np.ascontiguousarray(cmarg_sorted)
    best_score_per_eff = np.ascontiguousarray(score_sorted[:, 0].copy())

    # Per-eff feasibility marginal: BEST achievable constrained-stat roll at each eff
    # (max for a min constraint, min for a max constraint). Summed over a row's void
    # slots -> the optimistic reachable value, used to skip rows that provably can't
    # satisfy a hard constraint (fast infeasibility detection, no brute force).
    feas_per_eff = np.zeros((C, E))
    for ci, (s, sg, _) in enumerate(cons):
        feas_per_eff[ci] = cmarg[ci].max(1) if sg > 0 else cmarg[ci].min(1)

    dmax_s = dmin_s = None
    dmax_emax = dmin_emin = None
    if comp is not None:
        # Dep-tracking width: the simple composites (HPR/EHP/EHPR) keep the fixed
        # width 4 expected by _solve_row_comp/_node_ub; the spell composite uses its
        # actual dep count (32/36) consumed by _solve_row_spell.
        W = nd if is_spell else 4
        dmax_s = np.zeros((W, E, N)); dmin_s = np.zeros((W, E, N))
        for d in range(nd):
            dmax_s[d] = np.take_along_axis(dmax_e[d], order, axis=1)
            dmin_s[d] = np.take_along_axis(dmin_e[d], order, axis=1)
        dmax_s = np.ascontiguousarray(dmax_s); dmin_s = np.ascontiguousarray(dmin_s)
        # per-eff favourable extremes (max of each dep's max-roll marginal, min of each
        # dep's min-roll marginal) — drive both the per-row bound and the node-local UB.
        # Spell is monotone increasing so only dmax_emax matters there; dmin_emin is
        # still built (unused by the spell kernel but harmless / kept for symmetry).
        dmax_emax = np.zeros((W, E)); dmin_emin = np.zeros((W, E))
        dmax_emax[:nd] = dmax_e.max(2); dmin_emin[:nd] = dmin_e.min(2)
        dmax_emax = np.ascontiguousarray(dmax_emax); dmin_emin = np.ascontiguousarray(dmin_emin)

    cstat_idx = np.array([c[0] for c in cons], np.int64)
    cstat_max = sign > 0    # True -> use base_max column, else base_min
    pre = []
    all_bounds = []; all_n = []; all_m = []; all_infeas = []
    for n, b in enumerate(batches):
        M = b.ings_matrix.shape[0]
        if M == 0:
            pre.append(None); continue
        infeas_row = np.zeros(M, dtype=np.bool_)
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
        # Per-row LINEAR feasibility: a row is infeasible if even the optimistic
        # reachable value (best per-slot choices) cannot meet a hard min/max. Sound
        # (a necessary condition) and cheap vs the per-row solve. Only constraints with
        # a non-trivial threshold are checked (min<=0 / no upper are always satisfiable
        # given the search already excludes negative-only rows).
        for ci, (s, sg, thr_c) in enumerate(cons):
            opt = (imax[:, s] if sg > 0 else imin[:, s]).astype(np.float64)
            fpe = feas_per_eff[ci]
            for j in range(b.void_count):
                opt += fpe[veidx[:, j]]
            infeas_row |= (opt < thr_c) if sg > 0 else (opt > thr_c)
        comp_ub = None
        if comp is not None:
            # Per-row composite upper bound. d_hi[d] = max reachable max-roll total of
            # dep d (base_max + Σ_void per-eff max marginal); d_lo[d] = min reachable
            # min-roll total. Plug the EHP/HPR-favourable extremes into the (monotone)
            # leaf formula -> a valid over-estimate, sound for B&B pruning.
            d_hi = np.zeros((nd, M)); d_lo = np.zeros((nd, M))
            for d in range(nd):
                sidx = comp["deps"][d]
                d_hi[d] = imax[:, sidx].astype(np.float64)
                d_lo[d] = imin[:, sidx].astype(np.float64)
            for j in range(b.void_count):
                ix = veidx[:, j]
                for d in range(nd):
                    d_hi[d] += dmax_emax[d][ix]
                    d_lo[d] += dmin_emin[d][ix]
            if comp["type"] == _C_HPR:
                cub = _rtp_np(d_hi[0], d_hi[1]).astype(np.float64)
            elif comp["type"] == _C_EHP:
                cub = np.maximum(5.0, d_hi[0]) / _denom_min_np(d_hi[1], d_lo[2])
            elif comp["type"] == _C_EHPR:
                num_hi = _rtp_np(d_hi[0], d_hi[1]).astype(np.float64)
                cub = np.maximum(num_hi / _denom_min_np(d_hi[2], d_lo[3]),
                                 num_hi / _denom_max_np(d_lo[2], d_hi[3]))
            else:  # SPELL — evaluate the (monotone) corner at the favourable max box
                cub = _spell_bounds_all(np.ascontiguousarray(d_hi.T),
                                        comp["spell_id"], comp["use_spell"], spell_ctx)
            comp_ub = comp["w"] * cub
            bound = bound + comp_ub
            # Composite feasibility: cub is the optimistic max composite for the row;
            # if it can't reach a required composite min, no build in the row qualifies.
            if comp["has_min"]:
                infeas_row |= cub < comp["min"]
        pre.append((M, base_score, imax, imin, veidx, comp_ub))
        all_bounds.append(bound)
        all_infeas.append(infeas_row)
        all_n.append(np.full(M, n, np.int64))
        all_m.append(np.arange(M, dtype=np.int64))
    bounds_all = np.concatenate(all_bounds)
    infeas_all = np.concatenate(all_infeas)
    n_all = np.concatenate(all_n); m_all = np.concatenate(all_m)
    # Drop provably-infeasible rows BEFORE the search so an infeasible query collapses
    # fast (no per-row solve, no full scan); feasible rows keep their descending-bound
    # order. Empty -> INFEASIBLE immediately.
    order_rows = np.argsort(-bounds_all)
    order_rows = order_rows[~infeas_all[order_rows]]
    prep_s = time() - t

    total_rows = bounds_all.shape[0]
    n_feasible = order_rows.shape[0]
    if verbose:
        n_skipped = total_rows - n_feasible
        if n_skipped:
            print(f"[separable] feasibility: {n_skipped:,} of {total_rows:,} rows can't reach "
                  f"a hard min/max -> skipped (no per-row solve)")
        if n_feasible == 0:
            print("[separable] no row can satisfy the constraints -> INFEASIBLE")
        else:
            print(f"[separable] prep: {n_feasible:,} feasible meta-rows, "
                  f"top separable bound = {bounds_all[order_rows[0]]:,.0f}  (prep {prep_s:.2f}s)")
            print(f"[separable] search: branch & bound over {n_feasible:,} rows in descending "
                  f"bound order (stops when bound <= best)...")

    t2 = time()
    best = -1e18; best_rc = None; best_n = best_m = -1; solved = 0
    cutoff = None
    last_log = t2
    for oi in order_rows:
        if bounds_all[oi] <= best:
            cutoff = float(bounds_all[oi])
            break
        n = int(n_all[oi]); m = int(m_all[oi])
        _, base_score, imax, imin, veidx, comp_ub = pre[n]
        # constrained-stat base for THIS row only (min-constraint uses max-roll base,
        # max-constraint uses min-roll base).
        base_cval = np.where(cstat_max, imax[m, cstat_idx], imin[m, cstat_idx]).astype(np.float64)
        if comp is None:
            sc, choice = _solve_row(veidx[m], float(base_score[m]), base_cval,
                                    score_sorted, cmarg_sorted, best_score_per_eff,
                                    thr, sign, best)
        elif is_spell:
            ND = comp["n"]
            base_dmax = np.zeros(ND); base_dmin = np.zeros(ND)
            for d in range(ND):
                sidx = comp["deps"][d]
                base_dmax[d] = float(imax[m, sidx]); base_dmin[d] = float(imin[m, sidx])
            sc, choice = _solve_row_spell(
                veidx[m], float(base_score[m]), base_cval, base_dmin, base_dmax,
                score_sorted, cmarg_sorted, dmin_s, dmax_s, dmax_emax,
                ND, comp["spell_id"], comp["use_spell"], spell_ctx,
                best_score_per_eff, thr, sign, comp["w"],
                comp["has_min"], comp["min"], comp["has_max"], comp["max"], best)
        else:
            base_dmax = np.zeros(4); base_dmin = np.zeros(4)
            for d in range(comp["n"]):
                sidx = comp["deps"][d]
                base_dmax[d] = float(imax[m, sidx]); base_dmin[d] = float(imin[m, sidx])
            sc, choice = _solve_row_comp(
                veidx[m], float(base_score[m]), base_cval, base_dmin, base_dmax,
                score_sorted, cmarg_sorted, dmin_s, dmax_s, dmax_emax, dmin_emin,
                comp["n"], comp["type"], best_score_per_eff, thr, sign, comp["w"],
                comp["has_min"], comp["min"], comp["has_max"], comp["max"], best)
        solved += 1
        if sc > best:
            best = sc; best_rc = choice.copy(); best_n = n; best_m = m
        if verbose and time() - last_log >= 0.7:
            # B&B progress: best climbs, the cursor bound descends toward it; the loop
            # stops when they cross. headroom = how much score the survivors could add.
            cb = float(bounds_all[oi])
            if best > -1e17:
                tail = f"best = {best:,.0f} | cursor bound = {cb:,.0f} (headroom {cb - best:,.0f})"
            else:
                tail = f"best = (no feasible build yet) | cursor bound = {cb:,.0f}"
            print(f"    scanned {solved:,} rows | {tail}")
            last_log = time()
    search_s = time() - t2
    if verbose and n_feasible:
        pruned = (1.0 - solved / n_feasible) * 100.0 if n_feasible else 0.0
        cb = f"{cutoff:,.0f}" if cutoff is not None else "n/a (all rows scanned)"
        best_s = f"{best:,.0f}" if best > -1e17 else "INFEASIBLE"
        print(f"[separable] search: solved {solved:,}/{n_feasible:,} feasible rows, stopped "
              f"(cutoff bound {cb} <= best {best_s}) -> {pruned:.2f}% pruned  (search {search_s:.2f}s)")

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
        print(f"[separable] DONE -> score {best:,.1f}  "
              f"(load {load_s:.1f}s + prep {prep_s:.2f}s + search {search_s:.2f}s)")
    return {"build": build, "score": best, "status": "OPTIMAL", "meta_n": best_n,
            "rows_solved": solved, "recipe_raw": recipe_raw,
            "timings": {"load": load_s, "prep": prep_s, "search": search_s}}
