"""
search_engine.py
"""

import numpy as np
from numba import njit, prange
from numba import types
from queue import Queue
from threading import Thread
from time import time

from data.stats import (
    FORMULA_MUL_DIV_100, FORMULA_RAW_TO_PCT, FORMULA_EHP, FORMULA_EHPR,
    FORMULA_SPELL_DAMAGE_BASE,
)
from data.skillpoint_lookup import (
    SKP_HEADLINE_PCT, SKP_ELEMENT_PCT, SKP_PCT_BASE,
    SKP_DEF, SKP_AGI, SKP_MAX,
)
from data.spells import SPELL_MULTS, SPELL_USE_SPELL, SPELL_IGNORE_SPEED, SPELL_TOTAL_CONVERT

# Numba captures module-level numpy arrays as constants (read-only refs); the
# table is precomputed once at import time and never mutated, so this is safe
# even with cache=True.
_SKP_DEF_ROW = SKP_HEADLINE_PCT[SKP_DEF].copy()  # shape (151,) float64
_SKP_AGI_ROW = SKP_HEADLINE_PCT[SKP_AGI].copy()
# Per-element damage multiplier curves for spell formula. Shape (5, 151).
# Row order: str→earth, dex→thunder, int→water, def→fire, agi→air.
_SKP_ELEM_PCT = SKP_ELEMENT_PCT.copy()
# Base curve (no element multiplier). Shape (151,). Used for crit chance
# (skillPointsToPercentage(dex)) and str_boost (1 + skillPointsToPercentage(str)).
_SKP_PCT_BASE = SKP_PCT_BASE.copy()

# Spell metadata, captured as module-level numpy constants for numba access.
_SPELL_MULTS         = SPELL_MULTS.copy()          # int32 (N_SPELLS, 6)
_SPELL_USE_SPELL     = SPELL_USE_SPELL.copy()      # bool (N_SPELLS,)
_SPELL_IGNORE_SPEED  = SPELL_IGNORE_SPEED.copy()   # bool (N_SPELLS,)
_SPELL_TOTAL_CONVERT = SPELL_TOTAL_CONVERT.copy()  # float64 (N_SPELLS,)

# Build-context layout (must match BUILD_CTX_* in query/query.py).
# Stores BASE skill point counts (from non-crafted gear, supplied via _context).
# The build's crafted-ingredient skill point contribution is in the deps array
# itself; the kernel sums (base + crafted), clamps to [0, 150], then looks up
# the percentage curve.
_BCTX_BASE_STR = 0
_BCTX_BASE_DEX = 1
_BCTX_BASE_INT = 2
_BCTX_BASE_DEF = 3
_BCTX_BASE_AGI = 4
_BCTX_ATK_SPD  = 5
_BCTX_CRIT_PCT = 6
# Weapon's intrinsic per-element damage (post-powder, pre-roll). The spell
# kernel uses these for the m_n / m_e conversions; deps[0..5] (xDamRaw stat
# values) are treated as gear ID raws and added at step 5.2 only if the
# corresponding element is "present". Mirrors BUILD_CTX_WD_* in query.py.
_BCTX_WD_N = 7
_BCTX_WD_E = 8
_BCTX_WD_T = 9
_BCTX_WD_W = 10
_BCTX_WD_F = 11
_BCTX_WD_A = 12


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
    types.int32[::1],         # comp_dep_offset
    types.int32[::1],         # comp_dep_count
    types.int32[::1],         # comp_dep_indices (flat)
    types.int32[::1],         # comp_min
    types.int32[::1],         # comp_max
    types.Array(types.bool_, 1, "C"),  # comp_has_min
    types.Array(types.bool_, 1, "C"),  # comp_has_max
    types.float32[::1],       # comp_weight
    types.float64[::1],       # build_ctx (skp pcts, atk_spd, crit)
    types.int32[::1],         # round_offset (50 for req stats, 0 else)
    types.Array(types.bool_, 1, "C"),  # sp_score_active (True if SP stat)
    types.int32[::1],         # sp_score_ctx (ctx_base per stat)
    types.int32[::1],         # sp_score_req_idx (req's proj idx, -1 else)
    types.int32[::1],         # sp_score_req_base (user's req base, 0 else)
    types.float64[::1],       # deps_lo scratch (size ≥31, reused per spell call)
    types.float64[::1],       # deps_hi scratch
    types.Array(types.bool_, 1, "C"),  # useful_pos_eff (per ingredient, eff>0)
    types.Array(types.bool_, 1, "C"),  # useful_neg_eff (per ingredient, eff<=0)
)


@njit(cache=True)
def _compute_useful_masks(
    db_contrib_pos_mask, db_contrib_neg_mask,
    has_min_mask, has_max_mask, pos_weight_mask, neg_weight_mask,
):
    """
    Precompute per-(ingredient, sign(eff)) whether the ingredient can affect
    any constraint or weighted stat. The DFS used to rerun this O(S) scan
    at every node × every ingredient; the result depends only on (i, eff
    sign), so hoisting it here cuts an inner loop out of the hot path.

    `useful_pos_eff[i]` is True iff some stat s makes ingredient i useful
    when applied to a slot with eff > 0; `useful_neg_eff[i]` is the
    symmetric mask for eff <= 0 (matching the DFS's `eff_is_positive = eff > 0`).
    """
    N = db_contrib_pos_mask.shape[0]
    S = db_contrib_pos_mask.shape[1]
    useful_pos_eff = np.zeros(N, dtype=np.bool_)
    useful_neg_eff = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        for s in range(S):
            if db_contrib_pos_mask[i, s]:
                if has_min_mask[s] or pos_weight_mask[s]:
                    useful_pos_eff[i] = True
                if has_max_mask[s] or neg_weight_mask[s]:
                    useful_neg_eff[i] = True
            if db_contrib_neg_mask[i, s]:
                if has_max_mask[s] or neg_weight_mask[s]:
                    useful_pos_eff[i] = True
                if has_min_mask[s] or pos_weight_mask[s]:
                    useful_neg_eff[i] = True
            if useful_pos_eff[i] and useful_neg_eff[i]:
                break
    return useful_pos_eff, useful_neg_eff


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
# Spell damage formulas (generic — one evaluator for all 13 spells)
# ============================================================
# Wynnbuilder formula (`damage_calc.js::calculateSpellDamage`):
#
#   1. Conversions (per spell, mults = (m_n, m_e, m_t, m_w, m_f, m_a)):
#        damages[i]   = weapon_dam[i] * m_n/100             # neutral conv keeps element
#        damages[i>0] += m_i/100 * sum(weapon_dam)          # elem conv routes total -> i
#   2. Attack speed:    damages[i] *= atk_spd_mult          (skipped if ignore_speed)
#   3. % boost per element i:
#        boost[i] = 1 + skp[i]
#                     + (sdPct + damPct) / 100              # static_boost (mdPct for melee)
#                     + (eltDamPct + eltSdPct) / 100        # per-element % (mdPct variant absent here)
#                     + (i>0) * (rDamPct + rSdPct) / 100    # rainbow (rDamPct only for melee)
#        damages[i] *= boost[i]
#      where skp[i] = SKP_DAMAGE_MULT[i-1] * skillPointsToPercentage(skp_total[i-1])
#      i in {1..5} maps to {str->e, dex->t, int->w, def->f, agi->a}; skp[0] = 0.
#   4. Raw additions x total_convert (= sum(mults)/100):
#        For each element present[i] (weapon[i]>0 or m_i>0):
#          per_elem_raw[i] = eltSdRaw  (or eltMdRaw for melee)
#                            (eltDamRaw is the weapon damage in our model — NOT re-added)
#        global_raw = sdRaw (no damRaw in registry)
#        rainbow_raw = 0 (rSdRaw, rDamRaw not in registry — never present)
#        total_raw = (sum_present per_elem_raw[i]) + global_raw + rainbow_raw
#        sum_dmg += total_raw * total_convert
#   5. Per-element clamp at 0 (rare; we clamp the total instead — conservative for monotonicity).
#   6. Average damage (`optimize.js`):
#        critChance = skillPointsToPercentage(dex_total)            # 0..0.808
#        crit_mult  = 1 + critDamPct/100
#        avg_dmg    = sum_dmg * (str_boost + critChance * crit_mult)
#      where str_boost = 1 + skillPointsToPercentage(str_total).
#
# Monotonicity: every dep contributes monotone-increasingly to avg_dmg in
# practice (positive weapon damage, non-negative skp, sane boost ranges). We
# rely on this for B&B bounds: LB = formula(lo_corner), UB = formula(hi_corner).


@njit(cache=True)
def _eval_spell_corner_spell(deps, spell_id, ctx):
    """
    Evaluate average spell damage at one corner for a SPELL-mode spell
    (use_spell=True). `deps` is a length-31 float64 vector laid out per
    SPELL_SPELL_DEPS in data/spells.py.

    Weapon damage layout
    --------------------
    The weapon's intrinsic per-element damage (post-powder, pre-roll) lives
    in `ctx[_BCTX_WD_*]` — those are the values that get scaled by the spell
    multipliers (m_n, m_e, ...). `deps[0..5]` (xDamRaw stat values) are
    treated as **gear ID raws** and added at step 5.2 only when the
    corresponding element is "present" — exactly matching wynnbuilder's
    `calculateSpellDamage` (js/damage_calc.js). This separation keeps the
    spell formula faithful: ring/bracelet xDamRaw IDs no longer get
    incorrectly scaled by m_n + boost_t (which was a ~6× over-valuation
    of per-element raws relative to wynnbuilder).
    """
    mults = _SPELL_MULTS[spell_id]
    ignore_speed = _SPELL_IGNORE_SPEED[spell_id]
    total_convert = _SPELL_TOTAL_CONVERT[spell_id]

    # --- Weapon damage from ctx (intrinsic, post-powder) ---
    w_n = ctx[_BCTX_WD_N]; w_e = ctx[_BCTX_WD_E]; w_t = ctx[_BCTX_WD_T]
    w_w = ctx[_BCTX_WD_W]; w_f = ctx[_BCTX_WD_F]; w_a = ctx[_BCTX_WD_A]

    # --- Per-element xDamRaw IDs from gear (rings/etc) ---
    gx_n = deps[0]; gx_e = deps[1]; gx_t = deps[2]; gx_w = deps[3]; gx_f = deps[4]; gx_a = deps[5]
    p_n = deps[6]; p_e = deps[7]; p_t = deps[8]; p_w = deps[9]; p_f = deps[10]; p_a = deps[11]
    # Per-element SdPct (n,t omitted from registry -> treated as 0)
    sp_e = deps[12]; sp_w = deps[13]; sp_f = deps[14]; sp_a = deps[15]
    # Per-element SdRaw (n omitted; e,t,w,f,a in registry)
    sr_e = deps[16]; sr_t = deps[17]; sr_w = deps[18]; sr_f = deps[19]; sr_a = deps[20]
    # Globals + rainbow
    g_sd_pct = deps[21]; g_dam_pct = deps[22]; g_sd_raw = deps[23]
    r_dam_pct = deps[24]; r_sd_pct = deps[25]
    # Skill points (additive over user's base in ctx; clamped 0..150)
    s_str = int(deps[26] + ctx[_BCTX_BASE_STR])
    s_dex = int(deps[27] + ctx[_BCTX_BASE_DEX])
    s_int = int(deps[28] + ctx[_BCTX_BASE_INT])
    s_def = int(deps[29] + ctx[_BCTX_BASE_DEF])
    s_agi = int(deps[30] + ctx[_BCTX_BASE_AGI])
    if s_str < 0: s_str = 0
    elif s_str > SKP_MAX: s_str = SKP_MAX
    if s_dex < 0: s_dex = 0
    elif s_dex > SKP_MAX: s_dex = SKP_MAX
    if s_int < 0: s_int = 0
    elif s_int > SKP_MAX: s_int = SKP_MAX
    if s_def < 0: s_def = 0
    elif s_def > SKP_MAX: s_def = SKP_MAX
    if s_agi < 0: s_agi = 0
    elif s_agi > SKP_MAX: s_agi = SKP_MAX

    skp_e = _SKP_ELEM_PCT[0, s_str]   # earth from str
    skp_t = _SKP_ELEM_PCT[1, s_dex]   # thunder from dex
    skp_w = _SKP_ELEM_PCT[2, s_int]   # water from int
    skp_f = _SKP_ELEM_PCT[3, s_def]   # fire from def
    skp_a = _SKP_ELEM_PCT[4, s_agi]   # air from agi
    crit_chance = _SKP_PCT_BASE[s_dex]
    str_boost = 1.0 + _SKP_PCT_BASE[s_str]

    # --- Step 1+2: conversions x atk_spd ---
    m_n = mults[0] / 100.0
    m_e = mults[1] / 100.0
    m_t = mults[2] / 100.0
    m_w = mults[3] / 100.0
    m_f = mults[4] / 100.0
    m_a = mults[5] / 100.0
    total_w = w_n + w_e + w_t + w_w + w_f + w_a

    init_n = w_n * m_n
    init_e = w_e * m_n + (m_e * total_w if m_e > 0 else 0.0)
    init_t = w_t * m_n + (m_t * total_w if m_t > 0 else 0.0)
    init_w = w_w * m_n + (m_w * total_w if m_w > 0 else 0.0)
    init_f = w_f * m_n + (m_f * total_w if m_f > 0 else 0.0)
    init_a = w_a * m_n + (m_a * total_w if m_a > 0 else 0.0)

    atk_spd = 1.0 if ignore_speed else ctx[_BCTX_ATK_SPD]
    init_n *= atk_spd; init_e *= atk_spd; init_t *= atk_spd
    init_w *= atk_spd; init_f *= atk_spd; init_a *= atk_spd

    # --- Step 3: % boost per element ---
    static_boost = (g_sd_pct + g_dam_pct) / 100.0
    rainbow_boost = (r_dam_pct + r_sd_pct) / 100.0   # applied to non-neutral

    # Neutral: no skp, no rainbow, n SdPct absent -> 0
    boost_n = 1.0 + static_boost + p_n / 100.0
    boost_e = 1.0 + skp_e + static_boost + (p_e + sp_e) / 100.0 + rainbow_boost
    boost_t = 1.0 + skp_t + static_boost + (p_t + 0.0) / 100.0 + rainbow_boost   # tSdPct absent
    boost_w = 1.0 + skp_w + static_boost + (p_w + sp_w) / 100.0 + rainbow_boost
    boost_f = 1.0 + skp_f + static_boost + (p_f + sp_f) / 100.0 + rainbow_boost
    boost_a = 1.0 + skp_a + static_boost + (p_a + sp_a) / 100.0 + rainbow_boost

    sum_dmg = (init_n * boost_n + init_e * boost_e + init_t * boost_t
               + init_w * boost_w + init_f * boost_f + init_a * boost_a)

    # --- Step 4: raw additions (gated by per-element presence) ---
    # present[i] = (weapon[i] > 0) or (mults[i] > 0). Per-element raws are
    # eSdRaw + eDamRaw etc. (gx_*); n has no nSdRaw in the registry but does
    # have nDamRaw via gx_n.
    raw_sum = 0.0
    if (m_n > 0.0) and (w_n > 0.0): raw_sum += gx_n
    if (w_e > 0.0) or (m_e > 0.0): raw_sum += sr_e + gx_e
    if (w_t > 0.0) or (m_t > 0.0): raw_sum += sr_t + gx_t
    if (w_w > 0.0) or (m_w > 0.0): raw_sum += sr_w + gx_w
    if (w_f > 0.0) or (m_f > 0.0): raw_sum += sr_f + gx_f
    if (w_a > 0.0) or (m_a > 0.0): raw_sum += sr_a + gx_a
    raw_sum += g_sd_raw      # global (sdRaw + damRaw -> damRaw absent)
    sum_dmg += raw_sum * total_convert

    # --- Step 5: clamp + Step 6: average factor ---
    if sum_dmg < 0.0:
        sum_dmg = 0.0
    crit_mult = 1.0 + ctx[_BCTX_CRIT_PCT] / 100.0
    return sum_dmg * (str_boost + crit_chance * crit_mult)


@njit(cache=True)
def _eval_spell_corner_melee(deps, spell_id, ctx):
    """
    Same as `_eval_spell_corner_spell` but for MELEE-mode spells (use_spell=False).
    `deps` length 27, layout per SPELL_MELEE_DEPS in data/spells.py.

    Like the spell-mode kernel, weapon damage comes from `ctx[_BCTX_WD_*]`
    (set by search pipeline from recipe.weapon_dam_neutral or by an external
    caller that knows the equipped weapon). `deps[0..5]` (xDamRaw stat values)
    are gear ID raws added at step 5.2 only when the corresponding element
    is "present".
    """
    mults = _SPELL_MULTS[spell_id]
    ignore_speed = _SPELL_IGNORE_SPEED[spell_id]
    total_convert = _SPELL_TOTAL_CONVERT[spell_id]

    # Weapon damage from ctx (intrinsic, post-powder).
    w_n = ctx[_BCTX_WD_N]; w_e = ctx[_BCTX_WD_E]; w_t = ctx[_BCTX_WD_T]
    w_w = ctx[_BCTX_WD_W]; w_f = ctx[_BCTX_WD_F]; w_a = ctx[_BCTX_WD_A]

    # Per-element xDamRaw IDs from gear (rings/etc).
    gx_n = deps[0]; gx_e = deps[1]; gx_t = deps[2]; gx_w = deps[3]; gx_f = deps[4]; gx_a = deps[5]
    p_n = deps[6]; p_e = deps[7]; p_t = deps[8]; p_w = deps[9]; p_f = deps[10]; p_a = deps[11]
    # Per-element MdRaw — all 6 in registry
    mr_n = deps[12]; mr_e = deps[13]; mr_t = deps[14]; mr_w = deps[15]; mr_f = deps[16]; mr_a = deps[17]
    g_md_pct = deps[18]; g_dam_pct = deps[19]; g_md_raw = deps[20]
    r_dam_pct = deps[21]   # rMdPct absent -> no rainbow Md %

    s_str = int(deps[22] + ctx[_BCTX_BASE_STR])
    s_dex = int(deps[23] + ctx[_BCTX_BASE_DEX])
    s_int = int(deps[24] + ctx[_BCTX_BASE_INT])
    s_def = int(deps[25] + ctx[_BCTX_BASE_DEF])
    s_agi = int(deps[26] + ctx[_BCTX_BASE_AGI])
    if s_str < 0: s_str = 0
    elif s_str > SKP_MAX: s_str = SKP_MAX
    if s_dex < 0: s_dex = 0
    elif s_dex > SKP_MAX: s_dex = SKP_MAX
    if s_int < 0: s_int = 0
    elif s_int > SKP_MAX: s_int = SKP_MAX
    if s_def < 0: s_def = 0
    elif s_def > SKP_MAX: s_def = SKP_MAX
    if s_agi < 0: s_agi = 0
    elif s_agi > SKP_MAX: s_agi = SKP_MAX

    skp_e = _SKP_ELEM_PCT[0, s_str]
    skp_t = _SKP_ELEM_PCT[1, s_dex]
    skp_w = _SKP_ELEM_PCT[2, s_int]
    skp_f = _SKP_ELEM_PCT[3, s_def]
    skp_a = _SKP_ELEM_PCT[4, s_agi]
    crit_chance = _SKP_PCT_BASE[s_dex]
    str_boost = 1.0 + _SKP_PCT_BASE[s_str]

    m_n = mults[0] / 100.0
    m_e = mults[1] / 100.0
    m_t = mults[2] / 100.0
    m_w = mults[3] / 100.0
    m_f = mults[4] / 100.0
    m_a = mults[5] / 100.0
    total_w = w_n + w_e + w_t + w_w + w_f + w_a

    init_n = w_n * m_n
    init_e = w_e * m_n + (m_e * total_w if m_e > 0 else 0.0)
    init_t = w_t * m_n + (m_t * total_w if m_t > 0 else 0.0)
    init_w = w_w * m_n + (m_w * total_w if m_w > 0 else 0.0)
    init_f = w_f * m_n + (m_f * total_w if m_f > 0 else 0.0)
    init_a = w_a * m_n + (m_a * total_w if m_a > 0 else 0.0)

    atk_spd = 1.0 if ignore_speed else ctx[_BCTX_ATK_SPD]
    init_n *= atk_spd; init_e *= atk_spd; init_t *= atk_spd
    init_w *= atk_spd; init_f *= atk_spd; init_a *= atk_spd

    static_boost = (g_md_pct + g_dam_pct) / 100.0
    rainbow_boost = r_dam_pct / 100.0

    boost_n = 1.0 + static_boost + p_n / 100.0   # nMdPct absent -> 0
    boost_e = 1.0 + skp_e + static_boost + p_e / 100.0 + rainbow_boost   # eMdPct absent
    boost_t = 1.0 + skp_t + static_boost + p_t / 100.0 + rainbow_boost
    boost_w = 1.0 + skp_w + static_boost + p_w / 100.0 + rainbow_boost
    boost_f = 1.0 + skp_f + static_boost + p_f / 100.0 + rainbow_boost
    boost_a = 1.0 + skp_a + static_boost + p_a / 100.0 + rainbow_boost

    sum_dmg = (init_n * boost_n + init_e * boost_e + init_t * boost_t
               + init_w * boost_w + init_f * boost_f + init_a * boost_a)

    # Per-element raws: mdRaw (mr_*) + xDamRaw IDs (gx_*), gated by present.
    # Neutral is gated by m_n>0 AND w_n>0 (matches WB's `present=[false]*6`
    # short-circuit when neutral_convert == 0).
    raw_sum = 0.0
    if (m_n > 0.0) and (w_n > 0.0): raw_sum += mr_n + gx_n
    if (w_e > 0.0) or (m_e > 0.0): raw_sum += mr_e + gx_e
    if (w_t > 0.0) or (m_t > 0.0): raw_sum += mr_t + gx_t
    if (w_w > 0.0) or (m_w > 0.0): raw_sum += mr_w + gx_w
    if (w_f > 0.0) or (m_f > 0.0): raw_sum += mr_f + gx_f
    if (w_a > 0.0) or (m_a > 0.0): raw_sum += mr_a + gx_a
    raw_sum += g_md_raw
    sum_dmg += raw_sum * total_convert

    if sum_dmg < 0.0:
        sum_dmg = 0.0
    crit_mult = 1.0 + ctx[_BCTX_CRIT_PCT] / 100.0
    return sum_dmg * (str_boost + crit_chance * crit_mult)


# ============================================================
# Adapters — read deps from the appropriate per-call-site source(s)
# and dispatch to the right corner evaluator. Each adapter returns the
# (LB, UB) integer pair the search engine consumes.
# ============================================================

@njit(cache=True)
def _spell_leaf(spell_id, off, di, mins, maxs, ctx, deps_lo, deps_hi):
    """Bounds at the leaf — read deps from the build's roll min/max arrays.

    `deps_lo` and `deps_hi` are caller-provided scratch buffers of length ≥31
    (the largest spell-formula dep count). Pre-allocating once per m's DFS and
    reusing across all spell calls avoids ~2M np.empty() per batch on
    composite-heavy queries (mage_meteor: 31 deps × 2 arrays per UB/leaf call).
    """
    if _SPELL_USE_SPELL[spell_id]:
        for i in range(31):
            idx = di[off + i]
            deps_lo[i] = mins[idx]
            deps_hi[i] = maxs[idx]
        lb = _eval_spell_corner_spell(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_spell(deps_hi, spell_id, ctx)
    else:
        for i in range(27):
            idx = di[off + i]
            deps_lo[i] = mins[idx]
            deps_hi[i] = maxs[idx]
        lb = _eval_spell_corner_melee(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_melee(deps_hi, spell_id, ctx)
    return np.int64(lb), np.int64(ub)


@njit(cache=True)
def _spell_dfs_ub(spell_id, off, di, cur, flb, fub, nd, ctx, deps_lo, deps_hi):
    """
    Bounds at the dfs UB site — each dep value is `cur[idx] + future_lb` (for
    LB) or `cur[idx] + future_ub` (for UB). `nd` is the next-depth index into
    the future arrays. Buffers reused per caller's m loop (see _spell_leaf).
    """
    if _SPELL_USE_SPELL[spell_id]:
        for i in range(31):
            idx = di[off + i]
            base = cur[idx]
            deps_lo[i] = base + flb[nd, idx]
            deps_hi[i] = base + fub[nd, idx]
        lb = _eval_spell_corner_spell(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_spell(deps_hi, spell_id, ctx)
    else:
        for i in range(27):
            idx = di[off + i]
            base = cur[idx]
            deps_lo[i] = base + flb[nd, idx]
            deps_hi[i] = base + fub[nd, idx]
        lb = _eval_spell_corner_melee(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_melee(deps_hi, spell_id, ctx)
    return np.int64(lb), np.int64(ub)


@njit(cache=True)
def _spell_k2_ub(spell_id, off, di, after, sw, sb, ctx, deps_lo, deps_hi):
    """
    Bounds at the k=2 UB site — each dep value is `after[idx] + slot1_worst`
    (LB) or `after[idx] + slot1_best` (UB). Buffers reused per m's loop.
    """
    if _SPELL_USE_SPELL[spell_id]:
        for i in range(31):
            idx = di[off + i]
            base = after[idx]
            deps_lo[i] = base + sw[idx]
            deps_hi[i] = base + sb[idx]
        lb = _eval_spell_corner_spell(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_spell(deps_hi, spell_id, ctx)
    else:
        for i in range(27):
            idx = di[off + i]
            base = after[idx]
            deps_lo[i] = base + sw[idx]
            deps_hi[i] = base + sb[idx]
        lb = _eval_spell_corner_melee(deps_lo, spell_id, ctx)
        ub = _eval_spell_corner_melee(deps_hi, spell_id, ctx)
    return np.int64(lb), np.int64(ub)


# ============================================================
# Precompute per-meta-set bounds for B&B pruning
# ============================================================

@njit(cache=True)
def _precompute_bounds(db_stat_min, db_stat_max, meta_void_eff, void_count, dura_idx, round_offset,
                        future_max_ub, future_min_ub, future_max_lb, future_min_lb):
    """
    Build suffix-sum arrays capturing the best/worst additional contribution
    the remaining slots can add to current_max[s] and current_min[s].

    Convention: current_max[s] / current_min[s] track the TRUE reachable
    upper / lower bound on the build's total stat. With independent rolls
    per slot, that's sum-of-per-slot-max-contribution and sum-of-per-slot-
    min-contribution respectively. For a single slot d with effectiveness e:
      e >= 0 :  c_d_max = (db_stat_max * e + ro) // 100
                c_d_min = (db_stat_min * e + ro) // 100
      e <  0 :  c_d_max = (db_stat_min * e + ro) // 100   (less negative)
                c_d_min = (db_stat_max * e + ro) // 100   (more negative)
    `slot_best_max` / `slot_worst_max` are the max/min over ingredient choice
    of c_d_max; `slot_best_min` / `slot_worst_min` the same for c_d_min.

    Caller must provide four pre-allocated (void_count+1, S) int32 buffers
    (future_max_ub/min_ub/max_lb/min_lb). They are filled in-place; row k is
    zeroed (the empty-suffix sum). Pre-allocation outside the prange loop
    avoids ~M × 4 numpy allocations per batch — significant for big M.
    """
    S = db_stat_min.shape[1]
    N = db_stat_min.shape[0]
    k = void_count

    # Zero the "empty suffix" sentinel row (row k); the loop below writes the
    # rest based on row d+1, so seeding row k is enough.
    for s in range(S):
        future_max_ub[k, s] = 0
        future_min_ub[k, s] = 0
        future_max_lb[k, s] = 0
        future_min_lb[k, s] = 0

    # Walk slots from last to first; accumulate the per-slot best/worst
    # single-ingredient contribution into the suffix sum.
    for d in range(k - 1, -1, -1):
        eff = meta_void_eff[d]
        eff_pos = eff >= 0
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
                # Eff-sign-aware: db_stat_max ↔ db_stat_min swap when eff < 0,
                # because (large_db * negative_e) is more negative than
                # (small_db * negative_e).
                ro = round_offset[s]
                if eff_pos:
                    v_max = (db_stat_max[0, s] * eff + ro) // 100
                    v_min = (db_stat_min[0, s] * eff + ro) // 100
                else:
                    v_max = (db_stat_min[0, s] * eff + ro) // 100
                    v_min = (db_stat_max[0, s] * eff + ro) // 100
                slot_best_max = v_max
                slot_worst_max = v_max
                slot_best_min = v_min
                slot_worst_min = v_min
                for i in range(1, N):
                    if eff_pos:
                        v_max = (db_stat_max[i, s] * eff + ro) // 100
                        v_min = (db_stat_min[i, s] * eff + ro) // 100
                    else:
                        v_max = (db_stat_min[i, s] * eff + ro) // 100
                        v_min = (db_stat_max[i, s] * eff + ro) // 100
                    if v_max > slot_best_max:
                        slot_best_max = v_max
                    if v_max < slot_worst_max:
                        slot_worst_max = v_max
                    if v_min > slot_best_min:
                        slot_best_min = v_min
                    if v_min < slot_worst_min:
                        slot_worst_min = v_min

            future_max_ub[d, s] = future_max_ub[d + 1, s] + slot_best_max
            future_min_ub[d, s] = future_min_ub[d + 1, s] + slot_best_min
            future_max_lb[d, s] = future_max_lb[d + 1, s] + slot_worst_max
            future_min_lb[d, s] = future_min_lb[d + 1, s] + slot_worst_min


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
    comp_dep_offset,
    comp_dep_count,
    comp_dep_indices,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
    build_ctx,
    round_offset,
    sp_score_active,
    sp_score_ctx,
    sp_score_req_idx,
    sp_score_req_base,
    deps_lo,
    deps_hi,
    useful_pos_eff,
    useful_neg_eff,
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

            sm = max_v
            sn = min_v
            if sp_score_active[s]:
                # SP cap: useful = clamp(bonus, -ctx-req, SKP_MAX-ctx-req)
                # where req = max(user_req_base, craft's req in this leaf).
                # The req-allocation eats SP budget: player allocates max(reqs)
                # of skill points, which counts toward the cap (skillpoints.js
                # apply_to_fit / curve clamp). We apply directional req bounds:
                # max-track scoring uses min req (smallest cap denominator →
                # widest cap), min-track uses max req (smallest cap).
                ctx_b = sp_score_ctx[s]
                rb = sp_score_req_base[s]
                ri = sp_score_req_idx[s]
                if ri >= 0:
                    req_lo = current_min[ri]
                    req_hi = current_max[ri]
                else:
                    req_lo = 0
                    req_hi = 0
                if rb > req_lo: req_lo = rb
                if rb > req_hi: req_hi = rb
                hi_for_max = SKP_MAX - ctx_b - req_lo
                lo_for_max = -ctx_b - req_lo
                hi_for_min = SKP_MAX - ctx_b - req_hi
                lo_for_min = -ctx_b - req_hi
                if sm > hi_for_max: sm = hi_for_max
                if sm < lo_for_max: sm = lo_for_max
                if sn > hi_for_min: sn = hi_for_min
                if sn < lo_for_min: sn = lo_for_min
            score += weights[s] * sm * 0.99
            score += weights[s] * sn * 0.01

        # Composite stats: compute on the finalized build, check constraints,
        # add weighted contribution.
        for c in range(comp_count):
            off = comp_dep_offset[c]
            f = comp_formula[c]
            if f == FORMULA_MUL_DIV_100:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cmax = (np.int64(current_max[a]) * np.int64(current_max[b])) // 100
                cmin = (np.int64(current_min[a]) * np.int64(current_min[b])) // 100
            elif f == FORMULA_RAW_TO_PCT:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cmax = _raw_to_pct(current_max[a], current_max[b])
                cmin = _raw_to_pct(current_min[a], current_min[b])
            elif f == FORMULA_EHP:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                cmin, cmax = _ehp_bounds(
                    current_min[a], current_max[a],
                    current_min[b], current_max[b],
                    current_min[cc], current_max[cc],
                )
            elif f == FORMULA_EHPR:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                dd = comp_dep_indices[off + 3]
                cmin, cmax = _ehpr_bounds(
                    current_min[a], current_max[a],
                    current_min[b], current_max[b],
                    current_min[cc], current_max[cc],
                    current_min[dd], current_max[dd],
                )
            else:  # spell composite
                spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                cmin, cmax = _spell_leaf(
                    spell_id, off, comp_dep_indices,
                    current_min, current_max, build_ctx,
                    deps_lo, deps_hi,
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


    # Loop-invariant per depth — hoist outside the for-i loop so the compiler
    # doesn't re-fetch them every ingredient.
    eff = meta_void_eff[depth]
    eff_is_positive = eff > 0
    eff_pos = eff >= 0  # eff_pos: (db_max * e) is the upper extreme; eff_pos==False -> swap
    next_depth = depth + 1
    has_next_slot = next_depth < k
    same_eff_next = has_next_slot and (eff == meta_void_eff[next_depth])
    # Bind the right precomputed bitmask once per depth (eff sign is fixed
    # for this DFS frame). Inner loop becomes one bool lookup per ingredient
    # instead of an O(S) per-stat scan.
    if eff_is_positive:
        useful_for_eff = useful_pos_eff
    else:
        useful_for_eff = useful_neg_eff

    for i in range(start_index, db_count):

        # ============================================================
        # PRUNING
        # ============================================================

        if not useful_for_eff[i]:
            continue

        # ---- dura pruning ----
        if current_max[dura_idx] + db_stat_min[i, dura_idx] < min_vals[dura_idx]:
            continue

        # ============================================================
        #
        # ============================================================

        ingredients[depth] = i

        # Apply ingredient contribution. Eff-sign-aware swap so current_min /
        # current_max stay TRUE reachable extrema (see _precompute_bounds).
        for s in range(len(current_min)):
            if(s == dura_idx):
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

        # =========================
        # SCORE UB PRUNING (branch-and-bound) + HARD FEASIBILITY CHECK
        # =========================
        # Single fused pass over stats:
        #   1. Hard feasibility — if even the best-possible suffix can't make
        #      max_v ≥ min_vals (or worst-possible suffix can't keep min_v ≤
        #      max_vals), the entire subtree is infeasible and we skip it
        #      without bothering with composite UB or recursion. The leaf-only
        #      check used to catch this much later.
        #   2. UB score — same as before, optimistic per-stat upper bound on
        #      the final weighted score.
        ub_score = 0.0
        S = len(current_min)
        feasible = True
        for s in range(S):
            cmax = current_max[s]
            cmin = current_min[s]
            f_max_ub = future_max_ub[next_depth, s]
            f_min_ub = future_min_ub[next_depth, s]
            f_max_lb = future_max_lb[next_depth, s]
            f_min_lb = future_min_lb[next_depth, s]

            if has_min_mask[s] and cmax + f_max_ub < min_vals[s]:
                feasible = False
                break
            if has_max_mask[s] and cmin + f_min_lb > max_vals[s]:
                feasible = False
                break

            w = weights[s]
            if w != 0.0:
                if w > 0.0:
                    u_max = cmax + f_max_ub
                    u_min = cmin + f_min_ub
                else:
                    u_max = cmax + f_max_lb
                    u_min = cmin + f_min_lb
                if sp_score_active[s]:
                    # SP-cap-aware UB. To bound the cap conservatively in the
                    # right direction:
                    #   w>0 → max useful → use the SMALLEST possible req across
                    #     all future ingredient choices and rolls.
                    #   w<0 → min useful (max |w|*useful) → use the LARGEST.
                    ctx_b = sp_score_ctx[s]
                    rb = sp_score_req_base[s]
                    ri = sp_score_req_idx[s]
                    if w > 0.0:
                        if ri >= 0:
                            req_eff = current_min[ri] + future_min_lb[next_depth, ri]
                        else:
                            req_eff = 0
                    else:
                        if ri >= 0:
                            req_eff = current_max[ri] + future_max_ub[next_depth, ri]
                        else:
                            req_eff = 0
                    if rb > req_eff: req_eff = rb
                    hi_cap = SKP_MAX - ctx_b - req_eff
                    lo_cap = -ctx_b - req_eff
                    if u_max > hi_cap: u_max = hi_cap
                    if u_max < lo_cap: u_max = lo_cap
                    if u_min > hi_cap: u_min = hi_cap
                    if u_min < lo_cap: u_min = lo_cap
                ub_score += w * (u_max * 0.99 + u_min * 0.01)

        if not feasible:
            # Undo and skip — entire subtree can't satisfy hard constraints.
            for s in range(len(current_min)):
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

        # Composite UB: per-formula admissible bound on the final comp value,
        # using the (current_*[s] + future_*_*[next_depth, s]) range for each dep.
        # Loose vs true joint feasible set, but correct.
        for c in range(comp_count):
            w = comp_weight[c]
            if w == 0.0:
                continue
            off = comp_dep_offset[c]
            f = comp_formula[c]

            if f == FORMULA_MUL_DIV_100:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                pmax_lo, pmax_hi = _product_bounds_div100(
                    current_max[a] + future_max_lb[next_depth, a],
                    current_max[a] + future_max_ub[next_depth, a],
                    current_max[b] + future_max_lb[next_depth, b],
                    current_max[b] + future_max_ub[next_depth, b],
                )
                pmin_lo, pmin_hi = _product_bounds_div100(
                    current_min[a] + future_min_lb[next_depth, a],
                    current_min[a] + future_min_ub[next_depth, a],
                    current_min[b] + future_min_lb[next_depth, b],
                    current_min[b] + future_min_ub[next_depth, b],
                )
            elif f == FORMULA_RAW_TO_PCT:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                pmax_lo, pmax_hi = _raw_to_pct_bounds(
                    current_max[a] + future_max_lb[next_depth, a],
                    current_max[a] + future_max_ub[next_depth, a],
                    current_max[b] + future_max_lb[next_depth, b],
                    current_max[b] + future_max_ub[next_depth, b],
                )
                pmin_lo, pmin_hi = _raw_to_pct_bounds(
                    current_min[a] + future_min_lb[next_depth, a],
                    current_min[a] + future_min_ub[next_depth, a],
                    current_min[b] + future_min_lb[next_depth, b],
                    current_min[b] + future_min_ub[next_depth, b],
                )
            elif f == FORMULA_EHP:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                pmax_lo, pmax_hi = _ehp_bounds(
                    current_max[a] + future_max_lb[next_depth, a],
                    current_max[a] + future_max_ub[next_depth, a],
                    current_max[b] + future_max_lb[next_depth, b],
                    current_max[b] + future_max_ub[next_depth, b],
                    current_max[cc] + future_max_lb[next_depth, cc],
                    current_max[cc] + future_max_ub[next_depth, cc],
                )
                pmin_lo, pmin_hi = _ehp_bounds(
                    current_min[a] + future_min_lb[next_depth, a],
                    current_min[a] + future_min_ub[next_depth, a],
                    current_min[b] + future_min_lb[next_depth, b],
                    current_min[b] + future_min_ub[next_depth, b],
                    current_min[cc] + future_min_lb[next_depth, cc],
                    current_min[cc] + future_min_ub[next_depth, cc],
                )
            elif f == FORMULA_EHPR:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                dd = comp_dep_indices[off + 3]
                pmax_lo, pmax_hi = _ehpr_bounds(
                    current_max[a] + future_max_lb[next_depth, a],
                    current_max[a] + future_max_ub[next_depth, a],
                    current_max[b] + future_max_lb[next_depth, b],
                    current_max[b] + future_max_ub[next_depth, b],
                    current_max[cc] + future_max_lb[next_depth, cc],
                    current_max[cc] + future_max_ub[next_depth, cc],
                    current_max[dd] + future_max_lb[next_depth, dd],
                    current_max[dd] + future_max_ub[next_depth, dd],
                )
                pmin_lo, pmin_hi = _ehpr_bounds(
                    current_min[a] + future_min_lb[next_depth, a],
                    current_min[a] + future_min_ub[next_depth, a],
                    current_min[b] + future_min_lb[next_depth, b],
                    current_min[b] + future_min_ub[next_depth, b],
                    current_min[cc] + future_min_lb[next_depth, cc],
                    current_min[cc] + future_min_ub[next_depth, cc],
                    current_min[dd] + future_min_lb[next_depth, dd],
                    current_min[dd] + future_min_ub[next_depth, dd],
                )
            else:  # spell composite
                spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                pmax_lo, pmax_hi = _spell_dfs_ub(
                    spell_id, off, comp_dep_indices, current_max,
                    future_max_lb, future_max_ub, next_depth, build_ctx,
                    deps_lo, deps_hi,
                )
                pmin_lo, pmin_hi = _spell_dfs_ub(
                    spell_id, off, comp_dep_indices, current_min,
                    future_min_lb, future_min_ub, next_depth, build_ctx,
                    deps_lo, deps_hi,
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
                    ro = round_offset[s]
                    if eff_pos:
                        current_min[s] -= (db_stat_min[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_max[i, s] * eff + ro) // 100
                    else:
                        current_min[s] -= (db_stat_max[i, s] * eff + ro) // 100
                        current_max[s] -= (db_stat_min[i, s] * eff + ro) // 100
            continue

        # Permutation kill: only re-use the same start index when the next
        # void slot has the same effectiveness (slots are sorted desc by eff),
        # otherwise the slot is distinguishable and any ordering is allowed.
        # `same_eff_next` is precomputed above (loop-invariant); the k==6
        # special case (kept as in the original) means: at the deepest fan-out
        # level we always carry the start_index forward as a safety net.
        next_start = 0
        if k == 6 or same_eff_next:
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
            comp_dep_offset,
            comp_dep_count,
            comp_dep_indices,
            comp_min,
            comp_max,
            comp_has_min,
            comp_has_max,
            comp_weight,
            build_ctx,
            round_offset,
            sp_score_active,
            sp_score_ctx,
            sp_score_req_idx,
            sp_score_req_base,
            deps_lo,
            deps_hi,
            useful_pos_eff,
            useful_neg_eff,
        )

        # Undo
        for s in range(len(current_min)):
            if(s == dura_idx):
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
    comp_dep_offset,
    comp_dep_count,
    comp_dep_indices,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
    build_ctx,
    round_offset,
    sp_score_active,
    sp_score_ctx,
    sp_score_req_idx,
    sp_score_req_base,
):
    M = base_min_matrix.shape[0]
    S = base_min_matrix.shape[1]

    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, 1), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    # Shared best score across prange m's; racy but benign (see search_meta_batch).
    shared_best = np.array([init_best_score], dtype=np.float64)

    # Pre-allocate per-m scratch (final_min/max, deps_lo/hi). Each row is at
    # least 64 bytes so adjacent m's don't false-share.
    final_buf = np.empty((M, 2, S), dtype=np.int32)
    deps_buf = np.empty((M, 2, 31), dtype=np.float64)

    for m in prange(M):
        eff = void_eff_matrix[m, 0]
        eff_pos = eff >= 0
        sb_val = shared_best[0]
        local_best = init_best_score
        if sb_val > local_best:
            local_best = sb_val
        local_best_i = -1
        local_searched = 0

        # Per-iter storage so composite can reread dep values after the base loop.
        final_min = final_buf[m, 0]
        final_max = final_buf[m, 1]
        # Spell-eval scratch reused across every composite call this m.
        deps_lo = deps_buf[m, 0]
        deps_hi = deps_buf[m, 1]

        for i in range(db_count):
            feasible = True
            score = 0.0
            for s in range(S):
                if s == dura_idx:
                    min_v = base_min_matrix[m, s] + db_stat_min[i, s]
                    max_v = base_max_matrix[m, s] + db_stat_max[i, s]
                else:
                    # Eff-sign-aware: for negative eff, db_max contributes the
                    # MORE negative value (true min) and db_min the LESS negative
                    # (true max). Swap accordingly.
                    ro = round_offset[s]
                    if eff_pos:
                        min_v = base_min_matrix[m, s] + (db_stat_min[i, s] * eff + ro) // 100
                        max_v = base_max_matrix[m, s] + (db_stat_max[i, s] * eff + ro) // 100
                    else:
                        min_v = base_min_matrix[m, s] + (db_stat_max[i, s] * eff + ro) // 100
                        max_v = base_max_matrix[m, s] + (db_stat_min[i, s] * eff + ro) // 100

                final_min[s] = min_v
                final_max[s] = max_v

                if has_min_mask[s] and max_v < min_vals[s]:
                    feasible = False
                    break
                if has_max_mask[s] and min_v > max_vals[s]:
                    feasible = False
                    break

                sm = max_v
                sn = min_v
                if sp_score_active[s]:
                    # SP-cap clamp. The k1 leaf fills `final_*` progressively
                    # in stat order, so reading final_*[ri] for a forward req
                    # would be stale; recompute the req's per-leaf value
                    # inline using the same formula as above. Cheap because
                    # only ≤5 SP stats fire this branch.
                    ctx_b = sp_score_ctx[s]
                    rb = sp_score_req_base[s]
                    ri = sp_score_req_idx[s]
                    if ri >= 0:
                        if ri == dura_idx:
                            req_lo = base_min_matrix[m, ri] + db_stat_min[i, ri]
                            req_hi = base_max_matrix[m, ri] + db_stat_max[i, ri]
                        else:
                            ro_r = round_offset[ri]
                            if eff_pos:
                                req_lo = base_min_matrix[m, ri] + (db_stat_min[i, ri] * eff + ro_r) // 100
                                req_hi = base_max_matrix[m, ri] + (db_stat_max[i, ri] * eff + ro_r) // 100
                            else:
                                req_lo = base_min_matrix[m, ri] + (db_stat_max[i, ri] * eff + ro_r) // 100
                                req_hi = base_max_matrix[m, ri] + (db_stat_min[i, ri] * eff + ro_r) // 100
                    else:
                        req_lo = 0
                        req_hi = 0
                    if rb > req_lo: req_lo = rb
                    if rb > req_hi: req_hi = rb
                    hi_for_max = SKP_MAX - ctx_b - req_lo
                    lo_for_max = -ctx_b - req_lo
                    hi_for_min = SKP_MAX - ctx_b - req_hi
                    lo_for_min = -ctx_b - req_hi
                    if sm > hi_for_max: sm = hi_for_max
                    if sm < lo_for_max: sm = lo_for_max
                    if sn > hi_for_min: sn = hi_for_min
                    if sn < lo_for_min: sn = lo_for_min
                score += weights[s] * (sm * 0.99 + sn * 0.01)

            if not feasible:
                continue

            # Composite constraint + score
            for c in range(comp_count):
                off = comp_dep_offset[c]
                f = comp_formula[c]
                if f == FORMULA_MUL_DIV_100:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cmax = (np.int64(final_max[a]) * np.int64(final_max[b])) // 100
                    cmin = (np.int64(final_min[a]) * np.int64(final_min[b])) // 100
                elif f == FORMULA_RAW_TO_PCT:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cmax = _raw_to_pct(final_max[a], final_max[b])
                    cmin = _raw_to_pct(final_min[a], final_min[b])
                elif f == FORMULA_EHP:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cc = comp_dep_indices[off + 2]
                    cmin, cmax = _ehp_bounds(
                        final_min[a], final_max[a],
                        final_min[b], final_max[b],
                        final_min[cc], final_max[cc],
                    )
                elif f == FORMULA_EHPR:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cc = comp_dep_indices[off + 2]
                    dd = comp_dep_indices[off + 3]
                    cmin, cmax = _ehpr_bounds(
                        final_min[a], final_max[a],
                        final_min[b], final_max[b],
                        final_min[cc], final_max[cc],
                        final_min[dd], final_max[dd],
                    )
                else:  # spell composite
                    spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                    cmin, cmax = _spell_leaf(
                        spell_id, off, comp_dep_indices,
                        final_min, final_max, build_ctx,
                        deps_lo, deps_hi,
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

        if local_best_i >= 0:
            if local_best > shared_best[0]:
                shared_best[0] = local_best
            best_scores[m] = local_best
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
    comp_dep_offset,
    comp_dep_count,
    comp_dep_indices,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
    build_ctx,
    round_offset,
    sp_score_active,
    sp_score_ctx,
    sp_score_req_idx,
    sp_score_req_base,
):
    M = base_min_matrix.shape[0]
    S = base_min_matrix.shape[1]

    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, 2), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    # Shared best score across prange m's; racy but benign (see search_meta_batch).
    shared_best = np.array([init_best_score], dtype=np.float64)

    # Pre-allocate slot-1 bounds buffers (4×S int32 per m; ~144 bytes for S=9,
    # well above 64-byte cache lines so adjacent m rows don't share lines).
    slot1_buf = np.empty((M, 4, S), dtype=np.int32)
    # Per-m after-state + final-state for composite UB / leaf rereads.
    state_buf = np.empty((M, 4, S), dtype=np.int32)
    # Spell-eval scratch (max 31 deps).
    deps_buf = np.empty((M, 2, 31), dtype=np.float64)

    for m in prange(M):
        eff0 = void_eff_matrix[m, 0]
        eff1 = void_eff_matrix[m, 1]
        eff0_pos = eff0 >= 0
        eff1_pos = eff1 >= 0
        same_eff = (eff0 == eff1)

        # ---- Precompute slot-1 per-stat bounds (one-time O(N*S) per m).
        # Convention (eff-sign-aware):
        #   slot1_best_max[s]  = max over i of c_max(i)
        #   slot1_worst_max[s] = min over i of c_max(i)
        #   slot1_best_min[s]  = max over i of c_min(i)
        #   slot1_worst_min[s] = min over i of c_min(i)
        # where c_max / c_min are TRUE per-slot contribution extrema. For
        # eff1 < 0, db_max ↔ db_min swap (large db × negative e is more negative).
        slot1_best_max = slot1_buf[m, 0]
        slot1_best_min = slot1_buf[m, 1]
        slot1_worst_max = slot1_buf[m, 2]
        slot1_worst_min = slot1_buf[m, 3]

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
                ro = round_offset[s]
                if eff1_pos:
                    v_max = (db_stat_max[0, s] * eff1 + ro) // 100
                    v_min = (db_stat_min[0, s] * eff1 + ro) // 100
                else:
                    v_max = (db_stat_min[0, s] * eff1 + ro) // 100
                    v_min = (db_stat_max[0, s] * eff1 + ro) // 100
                bm = v_max
                wm = v_max
                bn = v_min
                wn = v_min
                for i in range(1, db_count):
                    if eff1_pos:
                        v_max = (db_stat_max[i, s] * eff1 + ro) // 100
                        v_min = (db_stat_min[i, s] * eff1 + ro) // 100
                    else:
                        v_max = (db_stat_min[i, s] * eff1 + ro) // 100
                        v_min = (db_stat_max[i, s] * eff1 + ro) // 100
                    if v_max > bm: bm = v_max
                    if v_max < wm: wm = v_max
                    if v_min > bn: bn = v_min
                    if v_min < wn: wn = v_min
            slot1_best_max[s] = bm
            slot1_worst_max[s] = wm
            slot1_best_min[s] = bn
            slot1_worst_min[s] = wn

        sb_val = shared_best[0]
        local_best = init_best_score
        if sb_val > local_best:
            local_best = sb_val
        local_best_i0 = -1
        local_best_i1 = -1
        local_searched = 0

        # Per-i0 after-state, needed again for composite UB.
        after_min_arr = state_buf[m, 0]
        after_max_arr = state_buf[m, 1]
        # Per-i1 final-state, needed again for composite leaf check + score.
        final_min_arr = state_buf[m, 2]
        final_max_arr = state_buf[m, 3]
        # Spell-eval scratch reused across every composite call this m.
        deps_lo = deps_buf[m, 0]
        deps_hi = deps_buf[m, 1]

        for i0 in range(db_count):
            # Apply i0: running state after placing first ingredient.
            # Compute UB on final score assuming slot 1 takes the best-possible ingredient.
            ub_score = 0.0
            for s in range(S):
                if s == dura_idx:
                    after_min = base_min_matrix[m, s] + db_stat_min[i0, s]
                    after_max = base_max_matrix[m, s] + db_stat_max[i0, s]
                else:
                    ro = round_offset[s]
                    if eff0_pos:
                        after_min = base_min_matrix[m, s] + (db_stat_min[i0, s] * eff0 + ro) // 100
                        after_max = base_max_matrix[m, s] + (db_stat_max[i0, s] * eff0 + ro) // 100
                    else:
                        after_min = base_min_matrix[m, s] + (db_stat_max[i0, s] * eff0 + ro) // 100
                        after_max = base_max_matrix[m, s] + (db_stat_min[i0, s] * eff0 + ro) // 100

                after_min_arr[s] = after_min
                after_max_arr[s] = after_max

                w = weights[s]
                if w != 0.0:
                    if w > 0.0:
                        u_max = after_max + slot1_best_max[s]
                        u_min = after_min + slot1_best_min[s]
                    else:
                        u_max = after_max + slot1_worst_max[s]
                        u_min = after_min + slot1_worst_min[s]
                    if sp_score_active[s]:
                        # SP-cap-aware UB. Need bounds on the OTHER slot's
                        # contribution to req. We don't have after_*[ri] yet
                        # (the loop just stores `after_*` for stat s), but
                        # base + db slot0's contribution is known. For slot 1,
                        # use slot1_best/worst on the req. For w>0 (max useful):
                        # use the SMALLEST possible req at slot 1; for w<0:
                        # use the LARGEST. Recompute slot 0's req inline.
                        ctx_b = sp_score_ctx[s]
                        rb = sp_score_req_base[s]
                        ri = sp_score_req_idx[s]
                        if ri >= 0:
                            # Slot-0 (i0) contribution to req at this m.
                            if ri == dura_idx:
                                req_after_min = base_min_matrix[m, ri] + db_stat_min[i0, ri]
                                req_after_max = base_max_matrix[m, ri] + db_stat_max[i0, ri]
                            else:
                                ro_r = round_offset[ri]
                                if eff0_pos:
                                    req_after_min = base_min_matrix[m, ri] + (db_stat_min[i0, ri] * eff0 + ro_r) // 100
                                    req_after_max = base_max_matrix[m, ri] + (db_stat_max[i0, ri] * eff0 + ro_r) // 100
                                else:
                                    req_after_min = base_min_matrix[m, ri] + (db_stat_max[i0, ri] * eff0 + ro_r) // 100
                                    req_after_max = base_max_matrix[m, ri] + (db_stat_min[i0, ri] * eff0 + ro_r) // 100
                            if w > 0.0:
                                req_eff = req_after_min + slot1_worst_min[ri]
                            else:
                                req_eff = req_after_max + slot1_best_max[ri]
                        else:
                            req_eff = 0
                        if rb > req_eff: req_eff = rb
                        hi_cap = SKP_MAX - ctx_b - req_eff
                        lo_cap = -ctx_b - req_eff
                        if u_max > hi_cap: u_max = hi_cap
                        if u_max < lo_cap: u_max = lo_cap
                        if u_min > hi_cap: u_min = hi_cap
                        if u_min < lo_cap: u_min = lo_cap
                    ub_score += w * (u_max * 0.99 + u_min * 0.01)

            # Composite UB: per-formula bound on [after_* + slot1_{worst,best}_*] rectangle.
            for c in range(comp_count):
                w = comp_weight[c]
                if w == 0.0:
                    continue
                off = comp_dep_offset[c]
                f = comp_formula[c]

                if f == FORMULA_MUL_DIV_100:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    pmax_lo, pmax_hi = _product_bounds_div100(
                        after_max_arr[a] + slot1_worst_max[a],
                        after_max_arr[a] + slot1_best_max[a],
                        after_max_arr[b] + slot1_worst_max[b],
                        after_max_arr[b] + slot1_best_max[b],
                    )
                    pmin_lo, pmin_hi = _product_bounds_div100(
                        after_min_arr[a] + slot1_worst_min[a],
                        after_min_arr[a] + slot1_best_min[a],
                        after_min_arr[b] + slot1_worst_min[b],
                        after_min_arr[b] + slot1_best_min[b],
                    )
                elif f == FORMULA_RAW_TO_PCT:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    pmax_lo, pmax_hi = _raw_to_pct_bounds(
                        after_max_arr[a] + slot1_worst_max[a],
                        after_max_arr[a] + slot1_best_max[a],
                        after_max_arr[b] + slot1_worst_max[b],
                        after_max_arr[b] + slot1_best_max[b],
                    )
                    pmin_lo, pmin_hi = _raw_to_pct_bounds(
                        after_min_arr[a] + slot1_worst_min[a],
                        after_min_arr[a] + slot1_best_min[a],
                        after_min_arr[b] + slot1_worst_min[b],
                        after_min_arr[b] + slot1_best_min[b],
                    )
                elif f == FORMULA_EHP:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cc = comp_dep_indices[off + 2]
                    pmax_lo, pmax_hi = _ehp_bounds(
                        after_max_arr[a] + slot1_worst_max[a],
                        after_max_arr[a] + slot1_best_max[a],
                        after_max_arr[b] + slot1_worst_max[b],
                        after_max_arr[b] + slot1_best_max[b],
                        after_max_arr[cc] + slot1_worst_max[cc],
                        after_max_arr[cc] + slot1_best_max[cc],
                    )
                    pmin_lo, pmin_hi = _ehp_bounds(
                        after_min_arr[a] + slot1_worst_min[a],
                        after_min_arr[a] + slot1_best_min[a],
                        after_min_arr[b] + slot1_worst_min[b],
                        after_min_arr[b] + slot1_best_min[b],
                        after_min_arr[cc] + slot1_worst_min[cc],
                        after_min_arr[cc] + slot1_best_min[cc],
                    )
                elif f == FORMULA_EHPR:
                    a = comp_dep_indices[off]
                    b = comp_dep_indices[off + 1]
                    cc = comp_dep_indices[off + 2]
                    dd = comp_dep_indices[off + 3]
                    pmax_lo, pmax_hi = _ehpr_bounds(
                        after_max_arr[a] + slot1_worst_max[a],
                        after_max_arr[a] + slot1_best_max[a],
                        after_max_arr[b] + slot1_worst_max[b],
                        after_max_arr[b] + slot1_best_max[b],
                        after_max_arr[cc] + slot1_worst_max[cc],
                        after_max_arr[cc] + slot1_best_max[cc],
                        after_max_arr[dd] + slot1_worst_max[dd],
                        after_max_arr[dd] + slot1_best_max[dd],
                    )
                    pmin_lo, pmin_hi = _ehpr_bounds(
                        after_min_arr[a] + slot1_worst_min[a],
                        after_min_arr[a] + slot1_best_min[a],
                        after_min_arr[b] + slot1_worst_min[b],
                        after_min_arr[b] + slot1_best_min[b],
                        after_min_arr[cc] + slot1_worst_min[cc],
                        after_min_arr[cc] + slot1_best_min[cc],
                        after_min_arr[dd] + slot1_worst_min[dd],
                        after_min_arr[dd] + slot1_best_min[dd],
                    )
                else:  # spell composite
                    spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                    pmax_lo, pmax_hi = _spell_k2_ub(
                        spell_id, off, comp_dep_indices, after_max_arr,
                        slot1_worst_max, slot1_best_max, build_ctx,
                        deps_lo, deps_hi,
                    )
                    pmin_lo, pmin_hi = _spell_k2_ub(
                        spell_id, off, comp_dep_indices, after_min_arr,
                        slot1_worst_min, slot1_best_min, build_ctx,
                        deps_lo, deps_hi,
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
                        # Eff-sign-aware swap per slot. Each slot's contribution
                        # extrema are independent; sum gives true reachable
                        # min/max of total stat.
                        ro = round_offset[s]
                        if eff0_pos:
                            c0_min = (db_stat_min[i0, s] * eff0 + ro) // 100
                            c0_max = (db_stat_max[i0, s] * eff0 + ro) // 100
                        else:
                            c0_min = (db_stat_max[i0, s] * eff0 + ro) // 100
                            c0_max = (db_stat_min[i0, s] * eff0 + ro) // 100
                        if eff1_pos:
                            c1_min = (db_stat_min[i1, s] * eff1 + ro) // 100
                            c1_max = (db_stat_max[i1, s] * eff1 + ro) // 100
                        else:
                            c1_min = (db_stat_max[i1, s] * eff1 + ro) // 100
                            c1_max = (db_stat_min[i1, s] * eff1 + ro) // 100
                        v_min = base_min_matrix[m, s] + c0_min + c1_min
                        v_max = base_max_matrix[m, s] + c0_max + c1_max

                    final_min_arr[s] = v_min
                    final_max_arr[s] = v_max

                    if has_min_mask[s] and v_max < min_vals[s]:
                        feasible = False
                        break
                    if has_max_mask[s] and v_min > max_vals[s]:
                        feasible = False
                        break

                    sm = v_max
                    sn = v_min
                    if sp_score_active[s]:
                        # SP-cap clamp; recompute req's per-leaf value inline
                        # since the k2 leaf fills final_*_arr forward.
                        ctx_b = sp_score_ctx[s]
                        rb = sp_score_req_base[s]
                        ri = sp_score_req_idx[s]
                        if ri >= 0:
                            if ri == dura_idx:
                                req_lo = (base_min_matrix[m, ri]
                                          + db_stat_min[i0, ri] + db_stat_min[i1, ri])
                                req_hi = (base_max_matrix[m, ri]
                                          + db_stat_max[i0, ri] + db_stat_max[i1, ri])
                            else:
                                ro_r = round_offset[ri]
                                if eff0_pos:
                                    c0_min_r = (db_stat_min[i0, ri] * eff0 + ro_r) // 100
                                    c0_max_r = (db_stat_max[i0, ri] * eff0 + ro_r) // 100
                                else:
                                    c0_min_r = (db_stat_max[i0, ri] * eff0 + ro_r) // 100
                                    c0_max_r = (db_stat_min[i0, ri] * eff0 + ro_r) // 100
                                if eff1_pos:
                                    c1_min_r = (db_stat_min[i1, ri] * eff1 + ro_r) // 100
                                    c1_max_r = (db_stat_max[i1, ri] * eff1 + ro_r) // 100
                                else:
                                    c1_min_r = (db_stat_max[i1, ri] * eff1 + ro_r) // 100
                                    c1_max_r = (db_stat_min[i1, ri] * eff1 + ro_r) // 100
                                req_lo = base_min_matrix[m, ri] + c0_min_r + c1_min_r
                                req_hi = base_max_matrix[m, ri] + c0_max_r + c1_max_r
                        else:
                            req_lo = 0
                            req_hi = 0
                        if rb > req_lo: req_lo = rb
                        if rb > req_hi: req_hi = rb
                        hi_for_max = SKP_MAX - ctx_b - req_lo
                        lo_for_max = -ctx_b - req_lo
                        hi_for_min = SKP_MAX - ctx_b - req_hi
                        lo_for_min = -ctx_b - req_hi
                        if sm > hi_for_max: sm = hi_for_max
                        if sm < lo_for_max: sm = lo_for_max
                        if sn > hi_for_min: sn = hi_for_min
                        if sn < lo_for_min: sn = lo_for_min
                    score += weights[s] * (sm * 0.99 + sn * 0.01)

                if not feasible:
                    continue

                # Composite constraint + score on the finalized k=2 build.
                for c in range(comp_count):
                    off = comp_dep_offset[c]
                    f = comp_formula[c]
                    if f == FORMULA_MUL_DIV_100:
                        a = comp_dep_indices[off]
                        b = comp_dep_indices[off + 1]
                        cmax = (np.int64(final_max_arr[a]) * np.int64(final_max_arr[b])) // 100
                        cmin = (np.int64(final_min_arr[a]) * np.int64(final_min_arr[b])) // 100
                    elif f == FORMULA_RAW_TO_PCT:
                        a = comp_dep_indices[off]
                        b = comp_dep_indices[off + 1]
                        cmax = _raw_to_pct(final_max_arr[a], final_max_arr[b])
                        cmin = _raw_to_pct(final_min_arr[a], final_min_arr[b])
                    elif f == FORMULA_EHP:
                        a = comp_dep_indices[off]
                        b = comp_dep_indices[off + 1]
                        cc = comp_dep_indices[off + 2]
                        cmin, cmax = _ehp_bounds(
                            final_min_arr[a], final_max_arr[a],
                            final_min_arr[b], final_max_arr[b],
                            final_min_arr[cc], final_max_arr[cc],
                        )
                    elif f == FORMULA_EHPR:
                        a = comp_dep_indices[off]
                        b = comp_dep_indices[off + 1]
                        cc = comp_dep_indices[off + 2]
                        dd = comp_dep_indices[off + 3]
                        cmin, cmax = _ehpr_bounds(
                            final_min_arr[a], final_max_arr[a],
                            final_min_arr[b], final_max_arr[b],
                            final_min_arr[cc], final_max_arr[cc],
                            final_min_arr[dd], final_max_arr[dd],
                        )
                    else:  # spell composite
                        spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                        cmin, cmax = _spell_leaf(
                            spell_id, off, comp_dep_indices,
                            final_min_arr, final_max_arr, build_ctx,
                            deps_lo, deps_hi,
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

        if local_best_i0 >= 0:
            if local_best > shared_best[0]:
                shared_best[0] = local_best
            best_scores[m] = local_best
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
    comp_dep_offset,
    comp_dep_count,
    comp_dep_indices,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
    build_ctx,
    round_offset,
    sp_score_active,
    sp_score_ctx,
    sp_score_req_idx,
    sp_score_req_base,
):
    M = ings_matrix.shape[0]
    k = void_count

    # Per-m output slots; no shared state between iterations.
    best_scores = np.full(M, -1e18, dtype=np.float64)
    best_solutions = np.zeros((M, k), dtype=np.int32)
    per_m_searched = np.zeros(M, dtype=np.int64)

    # Shared best score for cross-m BB pruning. Read/write is racy under
    # prange (no atomic CAS in CPU numba), but benign: each m's local DFS
    # uses its own best_score_ref so correctness is preserved. Worst case
    # is that shared_best gets clobbered to a lower value, costing some
    # pruning on subsequent m's but never losing a real solution.
    shared_best = np.array([init_best_score], dtype=np.float64)

    S = base_min_matrix.shape[1]

    # Per-(ingredient, eff sign) "useful" bitmask. Doesn't depend on m or
    # depth — same for every DFS call in this batch.
    useful_pos_eff, useful_neg_eff = _compute_useful_masks(
        db_contrib_pos_mask, db_contrib_neg_mask,
        has_min_mask, has_max_mask, pos_weight_mask, neg_weight_mask,
    )

    # Pre-allocate the bounds tensor (single big alloc instead of M×4 small
    # ones per batch). Stride per m = 4·(k+1)·S int32 ≈ 432 bytes for k=3,S=9
    # — well above 64-byte cache lines, so adjacent m's land on distinct lines
    # and false-sharing is not a concern.
    bounds_buf = np.zeros((M, 4, k + 1, S), dtype=np.int32)

    for m in prange(M):
        ingredients = np.zeros(k, dtype=np.int32)
        current_min = base_min_matrix[m].copy()
        current_max = base_max_matrix[m].copy()
        # Seed local best from the shared (cross-m) best so far.
        local_init = shared_best[0]
        if init_best_score > local_init:
            local_init = init_best_score
        best_score_ref = np.array([local_init], dtype=np.float64)
        best_solution_local = np.zeros(k, dtype=np.int32)
        local_searched = np.zeros(1, dtype=np.int64)
        seed_score = local_init

        future_max_ub = bounds_buf[m, 0]
        future_min_ub = bounds_buf[m, 1]
        future_max_lb = bounds_buf[m, 2]
        future_min_lb = bounds_buf[m, 3]
        _precompute_bounds(
            db_stat_min, db_stat_max, void_eff_matrix[m], k, dura_idx, round_offset,
            future_max_ub, future_min_ub, future_max_lb, future_min_lb,
        )

        # Spell-eval scratch reused across every spell helper call inside
        # this m's DFS (replaces ~thousands of np.empty(31) per m).
        deps_lo = np.empty(31, dtype=np.float64)
        deps_hi = np.empty(31, dtype=np.float64)

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
            comp_dep_offset,
            comp_dep_count,
            comp_dep_indices,
            comp_min,
            comp_max,
            comp_has_min,
            comp_has_max,
            comp_weight,
            build_ctx,
            round_offset,
            sp_score_active,
            sp_score_ctx,
            sp_score_req_idx,
            sp_score_req_base,
            deps_lo,
            deps_hi,
            useful_pos_eff,
            useful_neg_eff,
        )

        if best_score_ref[0] > seed_score:
            if best_score_ref[0] > shared_best[0]:
                shared_best[0] = best_score_ref[0]
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
# Search One Meta Batch (v2 — (m, i_0)-parallel)
# ============================================================
# `search_meta_batch` parallelizes only across m's. For low-M / high-k batches
# (META_5 with M=20, META_6 with M=1) that strands ~all of a 24-core CPU.
# This v2 fans out one DFS level deeper: the `prange` is over (m, i_0) so the
# outer loop has M*N work units. Same DFS body for depths 1..k-1 — we just
# inline the depth-0 logic and call the existing `dfs` with depth=1.
#
# Used for k>=3 batches (k=1 / k=2 keep their specialized paths). Memory use
# scales with M*N for the per-(m, i_0) result rows; for the largest k>=3
# batch (k=3, M~4k, N~100), that's ~3MB per result array — negligible.

@njit(parallel=True, cache=True)
def search_meta_batch_v2(
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
    comp_dep_offset,
    comp_dep_count,
    comp_dep_indices,
    comp_min,
    comp_max,
    comp_has_min,
    comp_has_max,
    comp_weight,
    build_ctx,
    round_offset,
    sp_score_active,
    sp_score_ctx,
    sp_score_req_idx,
    sp_score_req_base,
):
    M = ings_matrix.shape[0]
    k = void_count
    N = db_count
    S = base_min_matrix.shape[1]

    # Per-(m, i_0) output slots. We don't expect every (m, i_0) to yield a
    # build, but storing -1e18 sentinel + zero solution is cheap.
    best_scores_mi0 = np.full((M, N), -1e18, dtype=np.float64)
    best_solutions_mi0 = np.zeros((M, N, k), dtype=np.int32)
    per_mi0_searched = np.zeros((M, N), dtype=np.int64)

    # Cross-iteration best, racy-but-benign (same semantics as v1).
    shared_best = np.array([init_best_score], dtype=np.float64)

    # Per-(ingredient, eff sign) "useful" bitmask — same precompute as v1.
    useful_pos_eff, useful_neg_eff = _compute_useful_masks(
        db_contrib_pos_mask, db_contrib_neg_mask,
        has_min_mask, has_max_mask, pos_weight_mask, neg_weight_mask,
    )

    # Pre-allocate the bounds tensor once. Phase 1 fills bounds_buf[m] in a
    # parallel m-loop; Phase 2's (m, i_0) workers all read bounds_buf[m] but
    # only the slice for their m, so no false-sharing concerns.
    bounds_buf = np.zeros((M, 4, k + 1, S), dtype=np.int32)

    # Phase 1: precompute bounds for each m.
    for m in prange(M):
        future_max_ub_m = bounds_buf[m, 0]
        future_min_ub_m = bounds_buf[m, 1]
        future_max_lb_m = bounds_buf[m, 2]
        future_min_lb_m = bounds_buf[m, 3]
        _precompute_bounds(
            db_stat_min, db_stat_max, void_eff_matrix[m], k, dura_idx, round_offset,
            future_max_ub_m, future_min_ub_m, future_max_lb_m, future_min_lb_m,
        )

    # Phase 2: (m, i_0) parallel DFS.
    total_work = M * N
    for combined in prange(total_work):
        m = combined // N
        i_0 = combined % N

        # ----- Per-(m, i_0) state -----
        ingredients = np.zeros(k, dtype=np.int32)
        current_min = base_min_matrix[m].copy()
        current_max = base_max_matrix[m].copy()

        # Seed local best from shared (cross-iteration) best so far.
        local_init = shared_best[0]
        if init_best_score > local_init:
            local_init = init_best_score
        best_score_ref = np.array([local_init], dtype=np.float64)
        best_solution_local = np.zeros(k, dtype=np.int32)
        local_searched = np.zeros(1, dtype=np.int64)
        seed_score = local_init

        future_max_ub = bounds_buf[m, 0]
        future_min_ub = bounds_buf[m, 1]
        future_max_lb = bounds_buf[m, 2]
        future_min_lb = bounds_buf[m, 3]

        deps_lo = np.empty(31, dtype=np.float64)
        deps_hi = np.empty(31, dtype=np.float64)

        eff_0 = void_eff_matrix[m, 0]
        eff_0_is_positive = eff_0 > 0
        eff_0_pos = eff_0 >= 0

        # ----- Inlined depth=0 body of dfs (only for ingredient i = i_0) -----

        # Useful check via precomputed bitmask (was an O(S) scan per i_0).
        if eff_0_is_positive:
            if not useful_pos_eff[i_0]:
                continue
        else:
            if not useful_neg_eff[i_0]:
                continue

        # Dura prune
        if current_max[dura_idx] + db_stat_min[i_0, dura_idx] < min_vals[dura_idx]:
            continue

        ingredients[0] = i_0

        # Apply i_0 to slot 0. Eff-sign-aware swap so current_min / current_max
        # stay TRUE reachable extrema (see _precompute_bounds).
        for s in range(S):
            if s == dura_idx:
                current_min[s] += db_stat_min[i_0, s]
                current_max[s] += db_stat_max[i_0, s]
            else:
                ro = round_offset[s]
                if eff_0_pos:
                    current_min[s] += (db_stat_min[i_0, s] * eff_0 + ro) // 100
                    current_max[s] += (db_stat_max[i_0, s] * eff_0 + ro) // 100
                else:
                    current_min[s] += (db_stat_max[i_0, s] * eff_0 + ro) // 100
                    current_max[s] += (db_stat_min[i_0, s] * eff_0 + ro) // 100

        # UB check at depth 1 (next_depth = 1) — fused with hard feasibility
        # so we skip the (composite UB + recurse) work for infeasible builds.
        ub_score = 0.0
        feasible = True
        for s in range(S):
            cmax = current_max[s]
            cmin = current_min[s]
            f_max_ub = future_max_ub[1, s]
            f_min_ub = future_min_ub[1, s]
            f_max_lb = future_max_lb[1, s]
            f_min_lb = future_min_lb[1, s]

            if has_min_mask[s] and cmax + f_max_ub < min_vals[s]:
                feasible = False
                break
            if has_max_mask[s] and cmin + f_min_lb > max_vals[s]:
                feasible = False
                break

            w = weights[s]
            if w != 0.0:
                if w > 0.0:
                    u_max = cmax + f_max_ub
                    u_min = cmin + f_min_ub
                else:
                    u_max = cmax + f_max_lb
                    u_min = cmin + f_min_lb
                if sp_score_active[s]:
                    # SP-cap-aware UB at depth 1 — same logic as the dfs UB,
                    # but with the literal `1` (we're past slot 0, looking at
                    # the suffix from slot 1 onward).
                    ctx_b = sp_score_ctx[s]
                    rb = sp_score_req_base[s]
                    ri = sp_score_req_idx[s]
                    if w > 0.0:
                        if ri >= 0:
                            req_eff = current_min[ri] + future_min_lb[1, ri]
                        else:
                            req_eff = 0
                    else:
                        if ri >= 0:
                            req_eff = current_max[ri] + future_max_ub[1, ri]
                        else:
                            req_eff = 0
                    if rb > req_eff: req_eff = rb
                    hi_cap = SKP_MAX - ctx_b - req_eff
                    lo_cap = -ctx_b - req_eff
                    if u_max > hi_cap: u_max = hi_cap
                    if u_max < lo_cap: u_max = lo_cap
                    if u_min > hi_cap: u_min = hi_cap
                    if u_min < lo_cap: u_min = lo_cap
                ub_score += w * (u_max * 0.99 + u_min * 0.01)

        if not feasible:
            continue

        # Composite UB at depth 1 (mirrors the dfs composite UB block)
        for c in range(comp_count):
            w = comp_weight[c]
            if w == 0.0:
                continue
            off = comp_dep_offset[c]
            f = comp_formula[c]

            if f == FORMULA_MUL_DIV_100:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                pmax_lo, pmax_hi = _product_bounds_div100(
                    current_max[a] + future_max_lb[1, a],
                    current_max[a] + future_max_ub[1, a],
                    current_max[b] + future_max_lb[1, b],
                    current_max[b] + future_max_ub[1, b],
                )
                pmin_lo, pmin_hi = _product_bounds_div100(
                    current_min[a] + future_min_lb[1, a],
                    current_min[a] + future_min_ub[1, a],
                    current_min[b] + future_min_lb[1, b],
                    current_min[b] + future_min_ub[1, b],
                )
            elif f == FORMULA_RAW_TO_PCT:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                pmax_lo, pmax_hi = _raw_to_pct_bounds(
                    current_max[a] + future_max_lb[1, a],
                    current_max[a] + future_max_ub[1, a],
                    current_max[b] + future_max_lb[1, b],
                    current_max[b] + future_max_ub[1, b],
                )
                pmin_lo, pmin_hi = _raw_to_pct_bounds(
                    current_min[a] + future_min_lb[1, a],
                    current_min[a] + future_min_ub[1, a],
                    current_min[b] + future_min_lb[1, b],
                    current_min[b] + future_min_ub[1, b],
                )
            elif f == FORMULA_EHP:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                pmax_lo, pmax_hi = _ehp_bounds(
                    current_max[a] + future_max_lb[1, a],
                    current_max[a] + future_max_ub[1, a],
                    current_max[b] + future_max_lb[1, b],
                    current_max[b] + future_max_ub[1, b],
                    current_max[cc] + future_max_lb[1, cc],
                    current_max[cc] + future_max_ub[1, cc],
                )
                pmin_lo, pmin_hi = _ehp_bounds(
                    current_min[a] + future_min_lb[1, a],
                    current_min[a] + future_min_ub[1, a],
                    current_min[b] + future_min_lb[1, b],
                    current_min[b] + future_min_ub[1, b],
                    current_min[cc] + future_min_lb[1, cc],
                    current_min[cc] + future_min_ub[1, cc],
                )
            elif f == FORMULA_EHPR:
                a = comp_dep_indices[off]
                b = comp_dep_indices[off + 1]
                cc = comp_dep_indices[off + 2]
                dd = comp_dep_indices[off + 3]
                pmax_lo, pmax_hi = _ehpr_bounds(
                    current_max[a] + future_max_lb[1, a],
                    current_max[a] + future_max_ub[1, a],
                    current_max[b] + future_max_lb[1, b],
                    current_max[b] + future_max_ub[1, b],
                    current_max[cc] + future_max_lb[1, cc],
                    current_max[cc] + future_max_ub[1, cc],
                    current_max[dd] + future_max_lb[1, dd],
                    current_max[dd] + future_max_ub[1, dd],
                )
                pmin_lo, pmin_hi = _ehpr_bounds(
                    current_min[a] + future_min_lb[1, a],
                    current_min[a] + future_min_ub[1, a],
                    current_min[b] + future_min_lb[1, b],
                    current_min[b] + future_min_ub[1, b],
                    current_min[cc] + future_min_lb[1, cc],
                    current_min[cc] + future_min_ub[1, cc],
                    current_min[dd] + future_min_lb[1, dd],
                    current_min[dd] + future_min_ub[1, dd],
                )
            else:  # spell composite
                spell_id = f - FORMULA_SPELL_DAMAGE_BASE
                pmax_lo, pmax_hi = _spell_dfs_ub(
                    spell_id, off, comp_dep_indices, current_max,
                    future_max_lb, future_max_ub, 1, build_ctx,
                    deps_lo, deps_hi,
                )
                pmin_lo, pmin_hi = _spell_dfs_ub(
                    spell_id, off, comp_dep_indices, current_min,
                    future_min_lb, future_min_ub, 1, build_ctx,
                    deps_lo, deps_hi,
                )

            if w > 0.0:
                ub_score += w * (pmax_hi * 0.99 + pmin_hi * 0.01)
            else:
                ub_score += w * (pmax_lo * 0.99 + pmin_lo * 0.01)

        if ub_score <= best_score_ref[0]:
            continue

        # Permutation kill — at depth 0, next slot is depth 1; carry index
        # forward only when slots 0 and 1 share effectiveness (k>=2 here since
        # we route k=1/k=2 to specialized paths and v2 only handles k>=3).
        next_start = 0
        if k == 6 or (1 < k and eff_0 == void_eff_matrix[m, 1]):
            next_start = i_0

        # ----- Recurse for depths 1..k-1 -----
        dfs(
            1,                          # depth
            next_start,
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
            comp_dep_offset,
            comp_dep_count,
            comp_dep_indices,
            comp_min,
            comp_max,
            comp_has_min,
            comp_has_max,
            comp_weight,
            build_ctx,
            round_offset,
            sp_score_active,
            sp_score_ctx,
            sp_score_req_idx,
            sp_score_req_base,
            deps_lo,
            deps_hi,
            useful_pos_eff,
            useful_neg_eff,
        )

        # No undo of i_0 needed — current_min/max are this iteration's local copies.

        if best_score_ref[0] > seed_score:
            if best_score_ref[0] > shared_best[0]:
                shared_best[0] = best_score_ref[0]
            best_scores_mi0[m, i_0] = best_score_ref[0]
            for j in range(k):
                best_solutions_mi0[m, i_0, j] = best_solution_local[j]
        per_mi0_searched[m, i_0] = local_searched[0]

    # ----- Reduce: (m, i_0) -> global -----
    best_score = -1e18
    best_meta_index = -1
    best_solution_global = np.zeros(k, dtype=np.int32)
    for m in range(M):
        for i_0 in range(N):
            if best_scores_mi0[m, i_0] > best_score:
                best_score = best_scores_mi0[m, i_0]
                best_meta_index = m
                for j in range(k):
                    best_solution_global[j] = best_solutions_mi0[m, i_0, j]

    total = 0
    for m in range(M):
        for i_0 in range(N):
            total += per_mi0_searched[m, i_0]
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
            query.comp_dep_offset,
            query.comp_dep_count,
            query.comp_dep_indices,
            query.comp_min,
            query.comp_max,
            query.comp_has_min,
            query.comp_has_max,
            query.comp_weight,
            query.build_ctx,
            query.round_offset_proj,
            query.sp_score_active_proj,
            query.sp_score_ctx_proj,
            query.sp_score_req_idx_proj,
            query.sp_score_req_base_proj,
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
            query.comp_dep_offset,
            query.comp_dep_count,
            query.comp_dep_indices,
            query.comp_min,
            query.comp_max,
            query.comp_has_min,
            query.comp_has_max,
            query.comp_weight,
            query.build_ctx,
            query.round_offset_proj,
            query.sp_score_active_proj,
            query.sp_score_ctx_proj,
            query.sp_score_req_idx_proj,
            query.sp_score_req_base_proj,
        )
        total_searched[0] += added
        return score, meta_index, sol

    # k>=3: route through v2 which parallelizes across (m, i_0). Tried v3
    # ((m, i_0, i_1)) — measured no win because the extra parallelism breaks
    # intra-worker BB tightening (each (m, i_0) worker's `i_1` loop benefits
    # from local_best updates inside its own dfs, which v3 fragments away).
    return search_meta_batch_v2(
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
        query.comp_dep_offset,
        query.comp_dep_count,
        query.comp_dep_indices,
        query.comp_min,
        query.comp_max,
        query.comp_has_min,
        query.comp_has_max,
        query.comp_weight,
        query.build_ctx,
        query.round_offset_proj,
        query.sp_score_active_proj,
        query.sp_score_ctx_proj,
        query.sp_score_req_idx_proj,
        query.sp_score_req_base_proj,
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
    semi_cull_budget_s=0.0,
    semi_cull_max_n=5,
):
    """
    Overlap meta-set loading with searching. A background thread loads and
    refines each META_n file in priority order, pushing ready batches onto a
    queue; the main thread searches them as they arrive. Wall-clock = max(load
    time, search time) instead of their sum.

    Cull policy per META_n:
      - n <= max_cull        : full Pareto cull (no time limit).
      - max_cull < n <= semi_cull_max_n : semi-cull with `semi_cull_budget_s`
        wall-clock budget — the block-parallel cull stops after the budget
        and keeps unprocessed rows (Pareto SUPERSET, no optimum lost).
      - n > semi_cull_max_n  : skipped, all rows passed through.

    `semi_cull_budget_s` defaults to 0 (DISABLED). Empirically:
      - On medium META (~1-2M rows) it removes 4-5% of rows in ~1s and is
        a small net win.
      - On huge META (5M+ rows) the budget exhausts before meaningful kills
        and the parallel cull steals cores from the concurrent parallel
        search, costing more wall-clock than it saves. Tested on Armouring
        meteor (META_5=5.5M): semi-cull added +17s vs disabled.
      Enable per-call (e.g. `semi_cull_budget_s=1.0`) for queries where
      META sizes are known to be moderate.
    """
    # Local import avoids a module-level cycle (meta_set_loader imports nothing
    # from search_engine, but search_engine doesn't want the refiner eagerly).
    from data.meta_set_loader import _load_cached_arrays, _refine_batch

    # When crafting a weapon directly, overlay recipe.weapon_dam_neutral onto
    # `ctx[BUILD_CTX_WD_N]` so the spell kernel can scale it by m_n / m_e
    # (instead of treating ingredient nDamRaw rolls as weapon damage). For
    # non-weapon recipes this is (0, 0) and we leave the ctx untouched —
    # main_build_temp.py-style flows pre-populate all 6 weapon-dam slots
    # from a fixed loaded weapon before calling search_pipelined.
    if recipe.weapon_dam_neutral and (recipe.weapon_dam_neutral[0]
                                      or recipe.weapon_dam_neutral[1]):
        query.build_ctx[_BCTX_WD_N] = (recipe.weapon_dam_neutral[0]
                                       + recipe.weapon_dam_neutral[1]) / 2.0

    dura_idx = query.dura_proj_idx
    if dura_idx == -1:
        print("Dura not found in query. Aborting.")
        return None
    elif query.min_proj[dura_idx] < 1:
        print("Min dura should be strictly positive. Aborting.")
        return None

    priorities = [5, 4, 3, 2, 1, 0]

    q = Queue(maxsize=2)

    # Per-thread timing accumulators.
    # `load_time_total` counts wall-clock the producer thread spent fetching +
    # refining batches (cache hit/miss, JSON parse, projection). It runs in
    # parallel with searching, so it is NOT additive with `search_time_total`.
    load_time_total = [0.0]    # mutable via list so the closure can write
    wait_time_total = 0.0      # consumer wait on q.get() = "stall on loader"
    search_time_total = 0.0    # compute time inside _dispatch_search + _update_best

    # Per-stage breakdown (seconds, summed over META_n).
    breakdown = {"read": 0.0, "prep": 0.0, "cull": 0.0, "rows_in": 0, "rows_out": 0}
    per_n_breakdown = []  # list of (n, read, prep, cull, rows_in, rows_out)

    def producer():
        try:
            for n in priorities:
                t_load = time()
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
                    # Decide cull mode for this META_n.
                    cull_budget = None
                    if not culling:
                        use_culling = False
                    elif n <= max_cull:
                        use_culling = True  # full cull
                    elif (semi_cull_budget_s and semi_cull_budget_s > 0
                          and n <= semi_cull_max_n):
                        use_culling = True  # semi-cull with budget
                        cull_budget = float(semi_cull_budget_s)
                    else:
                        use_culling = False

                    t_read = time()
                    ings, eff, stat_names, stat_min, stat_max = \
                        _load_cached_arrays(skill, n, base_path)
                    read_dt = time() - t_read

                    timings = {"prep": 0.0, "cull": 0.0}
                    batch = _refine_batch(
                        ings, eff, stat_names, stat_min, stat_max,
                        query, recipe, culling=use_culling,
                        cull_budget_s=cull_budget,
                        timings=timings,
                    )
                    breakdown["read"] += read_dt
                    breakdown["prep"] += timings.get("prep", 0.0)
                    breakdown["cull"] += timings.get("cull", 0.0)
                    breakdown["rows_in"] += int(ings.shape[0])
                    breakdown["rows_out"] += int(batch.ings_matrix.shape[0])
                    per_n_breakdown.append((
                        n, read_dt, timings.get("prep", 0.0), timings.get("cull", 0.0),
                        int(ings.shape[0]), int(batch.ings_matrix.shape[0]),
                    ))
                load_time_total[0] += time() - t_load
                q.put((n, batch))
        except Exception as e:
            q.put(("ERROR", e))
        finally:
            q.put(None)

    t_pipeline = time()
    t = Thread(target=producer, daemon=True)
    t.start()

    best_score = -1e18
    best_full_slots = None
    total_searched = np.array([0], dtype=np.int64)
    total_possibilities = 0

    while True:
        t_wait = time()
        item = q.get()
        wait_time_total += time() - t_wait
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

        elapsed = time() - start_time
        search_time_total += elapsed
        print(
            f"meta batch {6 - meta_batch.void_count}: "
            f"{len(meta_batch.ings_matrix)}, "
            f"time elapsed: {elapsed * 1000:.0f}ms"
        )
        total_possibilities += len(meta_batch.ings_matrix) * db.count ** meta_batch.void_count

    t.join()
    wall = time() - t_pipeline

    print()
    print("SEARCHED FINISHED")
    print(f"Total combinations : {total_possibilities:,}")
    print(f"Total evaluated : {total_searched[0]:,}")
    if total_possibilities > 0:
        print(f"Pruning efficiency : {(1 - total_searched[0] / total_possibilities) * 100:.2f}% skipped")
    print()
    # Load and search run in parallel — wall ≈ max(load, search) + small overhead.
    # `consumer waited` is time the search thread blocked on q.get() (i.e. loader
    # was the bottleneck for that stretch). High wait → loading is the long pole.
    print(f"Load   : {load_time_total[0]:.1f}s  (producer thread, parallel with search)")
    print(f"Search : {search_time_total:.1f}s  (compute, sum of batches)")
    print(f"Wall   : {wall:.1f}s  (consumer waited {wait_time_total:.1f}s on loader)")
    print(
        f"  load breakdown: read={breakdown['read']:.1f}s "
        f"prep={breakdown['prep']:.1f}s cull={breakdown['cull']:.1f}s "
        f"({breakdown['rows_in']:,} -> {breakdown['rows_out']:,} rows)"
    )
    for n, rd, pr, cu, ri, ro in per_n_breakdown:
        print(
            f"    META_{n}: read={rd*1000:.0f}ms prep={pr*1000:.0f}ms "
            f"cull={cu*1000:.0f}ms ({ri:,} -> {ro:,})"
        )
    return best_full_slots