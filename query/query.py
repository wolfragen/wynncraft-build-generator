"""
query.py

Handles stat query parsing and projection.

Design goals:
- Support all stats defined in data.stats
- Preserve existing mask semantics
- Expose full stat space arrays for filtering
- Provide projected stat space for search
- No dict lookups in hot path
"""

import numpy as np
from typing import NamedTuple, Optional, List

from data.stats import (
    STAT_INDEX,
    STAT_COUNT,
    REQ_STATS,
    REQ_STATS_IDX,
    DERIVED_DEPENDENCIES,
    DERIVED_FORMULA,
    DEFAULT_BASE,
    FORMULA_MUL_DIV_100,
    FORMULA_RAW_TO_PCT,
    FORMULA_EHP,
    FORMULA_EHPR,
    FORMULA_SPELL_DAMAGE_BASE,
)
from data.spells import ATK_SPEED_MULT, SPELL_COUNT
from data.skillpoint_lookup import SKP_MAX


_FORMULA_TAGS = {
    "mul_div_100": FORMULA_MUL_DIV_100,
    "raw_to_pct": FORMULA_RAW_TO_PCT,
    "ehp": FORMULA_EHP,
    "ehpr": FORMULA_EHPR,
}

# Arity per formula tag — used by the parser to validate dep counts.
# Variable-arity formulas (e.g., spell composites) declare their arity
# dynamically in DERIVED_DEPENDENCIES; the parser checks `len(deps) ==
# _FORMULA_ARITY[tag]` for fixed-arity formulas, and skips this check for
# spell composites (their arity is whatever DERIVED_DEPENDENCIES declares).
_FORMULA_ARITY = {
    FORMULA_MUL_DIV_100: 2,
    FORMULA_RAW_TO_PCT: 2,
    FORMULA_EHP: 3,
    FORMULA_EHPR: 4,
}


# Build context layout (numpy float64 array passed to numba kernels).
# Indices are stable; the spell evaluator reads via these constants.
#
# Skill point slots store the BASE skill point COUNT (0..150) from the user's
# non-crafted gear (supplied via `_context["str"]` etc.). The spell formula
# adds the build's crafted-ingredient contribution (from the deps array),
# clamps to [0, 150], and looks up the percentage curve at evaluation time.
# This is needed because skillPointsToPercentage is non-linear, so we can't
# precompute the percentage once at the user level.
BUILD_CTX_BASE_STR = 0    # base skill point count (clamped 0..150)
BUILD_CTX_BASE_DEX = 1
BUILD_CTX_BASE_INT = 2
BUILD_CTX_BASE_DEF = 3
BUILD_CTX_BASE_AGI = 4
BUILD_CTX_ATK_SPD  = 5    # baseDamageMultiplier[atk_speed]
BUILD_CTX_CRIT_PCT = 6    # crit damage % (default 0)
# Weapon's intrinsic per-element damage (post-powder, pre-roll). Set by the
# search pipeline from the recipe's healthOrDamage (when crafting a weapon)
# or from a fixed loaded weapon (when filling around an existing build, see
# `main_build_temp.py`). The spell kernel uses these for the m_n / m_e
# conversions; deps[0..5] (xDamRaw stat values) are treated as gear ID raws
# and added at step 5.2 only if the corresponding element is "present".
# Element order matches `damage_elements = ['n'].concat(skp_elements)` in
# wynnbuilder, i.e. N, E, T, W, F, A.
BUILD_CTX_WD_N = 7
BUILD_CTX_WD_E = 8
BUILD_CTX_WD_T = 9
BUILD_CTX_WD_W = 10
BUILD_CTX_WD_F = 11
BUILD_CTX_WD_A = 12
BUILD_CTX_SIZE = 13


def _build_ctx_array(ctx_dict):
    """
    Build the float64 build-context array consumed by spell evaluators.
    `ctx_dict` is the user's `_context` entry, e.g.
        {"str": 80, "dex": 0, "int": 100, "def": 0, "agi": 40,
         "atk_speed": "NORMAL", "crit_dam_pct": 0}
    All keys optional; defaults: skillpoints=0, atk_speed="NORMAL", crit=0.
    Skill points stored as raw counts (clamped to [0, 150]) — the kernel sums
    them with the build's crafted contribution before looking up the curve.
    """
    arr = np.zeros(BUILD_CTX_SIZE, dtype=np.float64)
    if ctx_dict:
        for skp_name, slot in (("str", BUILD_CTX_BASE_STR),
                               ("dex", BUILD_CTX_BASE_DEX),
                               ("int", BUILD_CTX_BASE_INT),
                               ("def", BUILD_CTX_BASE_DEF),
                               ("agi", BUILD_CTX_BASE_AGI)):
            count = int(ctx_dict.get(skp_name, 0))
            if count < 0:
                count = 0
            elif count > SKP_MAX:
                count = SKP_MAX
            arr[slot] = float(count)
        atk_speed = ctx_dict.get("atk_speed", "NORMAL")
        arr[BUILD_CTX_ATK_SPD] = float(ATK_SPEED_MULT.get(atk_speed, ATK_SPEED_MULT["NORMAL"]))
        arr[BUILD_CTX_CRIT_PCT] = float(ctx_dict.get("crit_dam_pct", 0))
        # weapon_dam: optional 6-tuple (n, e, t, w, f, a) for builds with a
        # fixed already-equipped weapon. Default 0 — search_pipelined will
        # overlay recipe.weapon_dam_neutral when crafting a weapon directly.
        weapon_dam = ctx_dict.get("weapon_dam")
        if weapon_dam:
            for i, slot in enumerate((BUILD_CTX_WD_N, BUILD_CTX_WD_E, BUILD_CTX_WD_T,
                                       BUILD_CTX_WD_W, BUILD_CTX_WD_F, BUILD_CTX_WD_A)):
                if i < len(weapon_dam):
                    arr[slot] = float(weapon_dam[i])
    else:
        arr[BUILD_CTX_ATK_SPD] = ATK_SPEED_MULT["NORMAL"]
    return arr


class Query(NamedTuple):
    search_for_inversion: bool
    item_type: Optional[str]
    skill: Optional[str]

    min_stats: np.ndarray
    max_stats: np.ndarray
    weights: np.ndarray

    has_min_mask: np.ndarray
    has_max_mask: np.ndarray

    base_min_stats_proj: np.ndarray
    base_max_stats_proj: np.ndarray

    active_indices: np.ndarray
    stat_count: int

    min_proj: np.ndarray
    max_proj: np.ndarray
    weights_proj: np.ndarray
    pos_weight_mask_proj: np.ndarray
    neg_weight_mask_proj: np.ndarray

    has_min_mask_proj: np.ndarray
    has_max_mask_proj: np.ndarray

    stat_index_keys_proj: List[str]
    req_mask_proj: np.ndarray
    req_idx: np.ndarray
    filter_mask: np.ndarray
    proj_stats_idx: np.ndarray

    # Per-stat rounding bias added before `// 100` in eff scaling. 50 for req
    # stats (matches wynnbuilder's `Math.round(value * eff_mult)` for itemIDs
    # in craft.js:464-468), 0 for everything else (matches its `Math.floor` for
    # rolled IDs in craft.js:487). Without this, `(v*eff)//100` undercounts
    # negative-half contributions by 1 per slot vs WB, making xReq max
    # constraints leak by up to ~6 units on 6-void builds.
    round_offset_proj: np.ndarray   # int32[stat_count]

    consumable: bool
    suggested_max_cull: int

    # Projected-space index of durability (crafted) or duration (consumable).
    # -1 if neither is active — search will abort in that case.
    dura_proj_idx: int

    # Build-context constants (skillpoint percentages, atk_speed mult, crit pct)
    # passed verbatim to numba kernels. See BUILD_CTX_* indices above.
    build_ctx: np.ndarray  # float64[BUILD_CTX_SIZE]

    # SP-cap-aware scoring metadata (replaces the older static skp_score_lo/hi
    # arrays). The cap on a SP stat's useful contribution depends on the
    # craft's matching skill requirement (player allocates max(reqs) of SP,
    # which counts toward the total → reduces the cap headroom for the bonus
    # score), so we look up the req's per-leaf value dynamically rather than
    # precomputing a static clamp.
    #
    #   sp_score_active_proj[s]  : True iff stat s is one of str/dex/int/def/agi
    #                              and should be scored with the SP cap.
    #   sp_score_ctx_proj[s]     : the user's ctx_base for that SP (from
    #                              `_context`), already clamped to [0, SKP_MAX].
    #   sp_score_req_idx_proj[s] : projected idx of the corresponding xReq
    #                              stat (e.g., dex → dexReq), or -1 if no
    #                              req is active in the query.
    #   sp_score_req_base_proj[s]: user-provided `base` (or `min_base`) on
    #                              that req stat — combined via MAX with the
    #                              craft's req in the cap calc. 0 if none.
    sp_score_active_proj: np.ndarray   # bool[stat_count]
    sp_score_ctx_proj: np.ndarray      # int32[stat_count]
    sp_score_req_idx_proj: np.ndarray  # int32[stat_count]
    sp_score_req_base_proj: np.ndarray # int32[stat_count]

    # Composite (derived) stats — Struct-of-Arrays for numba passage.
    # All indexed 0..comp_count-1. Dep indices are stored in a flat array,
    # variable-arity per composite via (offset, count) pairs.
    #   For composite c, deps live at comp_dep_indices[offset:offset+count],
    #   ordered per-formula's expected dep order (e.g., for ehp: hpBonus, def, agi).
    comp_count: int
    comp_formula: np.ndarray        # int32[N]
    comp_dep_offset: np.ndarray     # int32[N] — start in comp_dep_indices
    comp_dep_count: np.ndarray      # int32[N] — number of deps for composite c
    comp_dep_indices: np.ndarray    # int32[total_deps] — flat projected indices
    comp_min: np.ndarray            # int32[N]
    comp_max: np.ndarray            # int32[N]
    comp_has_min: np.ndarray        # bool[N]
    comp_has_max: np.ndarray        # bool[N]
    comp_weight: np.ndarray         # float32[N]


def build_query(
    user_json: dict,
    search_for_inversion: bool,
    item_type: Optional[str] = None,
    skill: Optional[str] = None,
    consumable: bool = False,
) -> Query:
    """
    Parse user query.

    Args:
        user_json: dict of stat constraints
        search_for_inversion: bool
        item_type: optional crafting profession filter
        skill: optional skill filter (reserved for later use)
    """

    # ------------------------------------------------------------
    # Full stat space storage (used by filter)
    # ------------------------------------------------------------
    min_stats = np.zeros(STAT_COUNT, dtype=np.int32)
    max_stats = np.zeros(STAT_COUNT, dtype=np.int32)
    weights = np.zeros(STAT_COUNT, dtype=np.float32)

    has_min_mask = np.zeros(STAT_COUNT, dtype=np.bool_)
    has_max_mask = np.zeros(STAT_COUNT, dtype=np.bool_)

    base_min_stats = np.zeros(STAT_COUNT, dtype=np.int32)
    base_max_stats = np.zeros(STAT_COUNT, dtype=np.int32)

    active_mask = np.zeros(STAT_COUNT, dtype=np.bool_)

    # Any stat with a constraint (min/max/weight) is used as an ingredient
    # filter by default; user can opt out per-stat via `"ingredient_filter": False`.
    should_filter = np.zeros(STAT_COUNT, dtype=np.bool_)
    # Tracks which stats had an explicit `ingredient_filter` in user_json, so
    # composite propagation doesn't override the user's explicit choice.
    explicit_filter_set = np.zeros(STAT_COUNT, dtype=np.bool_)
    # Tracks which stats had an explicit base override, so composite dep
    # activation doesn't overwrite it with DEFAULT_BASE.
    explicit_base_set = np.zeros(STAT_COUNT, dtype=np.bool_)

    # User-provided `base` for skill-req stats — captured separately so the
    # search treats it via MAX (not SUM) with the craft's req. The additive
    # `base_min/max_stats` slot for req stats stays at 0; the constraint
    # check applies to the craft alone (valid since `max(base, craft) ≤
    # max_constraint ⇔ base ≤ max_constraint AND craft ≤ max_constraint`,
    # and we validate the first conjunct upfront). Indexed by req stat
    # global idx (strReq..agiReq); 0 means "no user base".
    req_user_base_full = np.zeros(STAT_COUNT, dtype=np.int32)

    # ------------------------------------------------------------
    # Build context (skillpoints, atk speed, crit) — query-level constants
    # used by spell composites. Pulled out of user_json before stat parsing.
    # ------------------------------------------------------------
    build_ctx_dict = user_json.get("_context") if isinstance(user_json, dict) else None
    build_ctx_arr = _build_ctx_array(build_ctx_dict)

    # ------------------------------------------------------------
    # Pass 1: non-composite (base) stats
    # ------------------------------------------------------------
    composite_entries = []

    for stat_name, config in user_json.items():

        if stat_name == "_context":
            continue  # already consumed

        deps = DERIVED_DEPENDENCIES.get(stat_name)

        if deps is not None:
            # Defer composite handling to pass 2.
            composite_entries.append((stat_name, config, deps))
            continue

        idx = STAT_INDEX.get(stat_name)
        if idx is None:
            continue

        stat_min = config.get("min")
        stat_max = config.get("max")
        stat_weight = config.get("weight")

        if stat_min is not None:
            min_stats[idx] = stat_min
            has_min_mask[idx] = True
            active_mask[idx] = True

        if stat_max is not None:
            max_stats[idx] = stat_max
            has_max_mask[idx] = True
            active_mask[idx] = True

        if stat_weight is not None:
            weights[idx] = stat_weight
            active_mask[idx] = True

        if stat_min is not None or stat_max is not None or stat_weight is not None:
            should_filter[idx] = True

        stat_base = config.get("base")
        stat_min_base = config.get("min_base")
        stat_max_base = config.get("max_base")

        is_req_stat = stat_name in REQ_STATS

        if is_req_stat:
            # Skill-req stats use MAX semantic (not SUM) when combining the
            # user's base with the craft. Capture the base into a side array
            # and skip the additive base_*_stats writes — the constraint
            # check then applies to the craft alone (we validate the user's
            # base against any user-set max constraint below).
            user_req_base = 0
            if stat_base is not None:
                user_req_base = max(user_req_base, int(stat_base))
            if stat_min_base is not None:
                user_req_base = max(user_req_base, int(stat_min_base))
            if stat_max_base is not None:
                user_req_base = max(user_req_base, int(stat_max_base))
            if user_req_base > 0:
                req_user_base_full[idx] = user_req_base
                explicit_base_set[idx] = True
                if stat_max is not None and user_req_base > stat_max:
                    raise ValueError(
                        f"Query '{stat_name}': base ({user_req_base}) > max "
                        f"({stat_max}). The craft alone cannot drop the "
                        f"build's req below the user-supplied base."
                    )
        else:
            if stat_base is not None:
                base_min_stats[idx] = stat_base
                base_max_stats[idx] = stat_base
                explicit_base_set[idx] = True

            if stat_min_base is not None:
                base_min_stats[idx] = stat_min_base
                explicit_base_set[idx] = True

            if stat_max_base is not None:
                base_max_stats[idx] = stat_max_base
                explicit_base_set[idx] = True

        should_filter_stat = config.get("ingredient_filter")
        if should_filter_stat is not None:
            should_filter[idx] = bool(should_filter_stat)
            explicit_filter_set[idx] = True

    # ------------------------------------------------------------
    # Pass 2: composite (derived) stats
    # ------------------------------------------------------------
    comp_formulas = []
    comp_dep_names = []   # list of tuples, padded to MAX_COMP_ARITY with None
    comp_min_vals = []
    comp_max_vals = []
    comp_has_min_vals = []
    comp_has_max_vals = []
    comp_weight_vals = []

    for stat_name, config, deps in composite_entries:

        # Composite stats derive their base from their deps — accepting a
        # per-composite "base" would be ambiguous (which dep does it apply to?)
        # and silently ignored today. Reject explicitly to avoid surprises.
        for forbidden in ("base", "min_base", "max_base"):
            if forbidden in config:
                raise ValueError(
                    f"Composite stat '{stat_name}' does not accept '{forbidden}'. "
                    f"Set it on one of its dependencies ({', '.join(deps)}) instead."
                )

        formula_name = DERIVED_FORMULA.get(stat_name)
        # Tuple form (variable-arity formulas, e.g. ("spell", spell_id)).
        if isinstance(formula_name, tuple) and formula_name and formula_name[0] == "spell":
            spell_id = int(formula_name[1])
            if not (0 <= spell_id < SPELL_COUNT):
                raise ValueError(
                    f"Composite '{stat_name}': spell_id {spell_id} out of range "
                    f"(have {SPELL_COUNT} spells defined)."
                )
            formula_tag = FORMULA_SPELL_DAMAGE_BASE + spell_id
            # Spells declare arity via DERIVED_DEPENDENCIES — no further check.
        else:
            formula_tag = _FORMULA_TAGS.get(formula_name)
            if formula_tag is None:
                continue
            expected_arity = _FORMULA_ARITY[formula_tag]
            if len(deps) != expected_arity:
                raise ValueError(
                    f"Composite '{stat_name}' formula '{formula_name}' expects "
                    f"{expected_arity} deps, got {len(deps)}: {deps}"
                )

        # Activate dependency stats + apply DEFAULT_BASE if not overridden.
        for dep in deps:
            dep_idx = STAT_INDEX[dep]
            active_mask[dep_idx] = True
            if not explicit_base_set[dep_idx]:
                default_b = DEFAULT_BASE.get(dep)
                if default_b is not None:
                    base_min_stats[dep_idx] = default_b
                    base_max_stats[dep_idx] = default_b

        stat_min = config.get("min")
        stat_max = config.get("max")
        stat_weight = config.get("weight")

        has_min = stat_min is not None
        has_max = stat_max is not None
        has_weight = stat_weight is not None

        if not (has_min or has_max or has_weight):
            # No actual constraint — skip, composite is inert.
            continue

        # Propagate ingredient_filter to deps, respecting explicit dep overrides.
        comp_filter_raw = config.get("ingredient_filter")
        # Composite active → defaults to True, matching base-stat semantics.
        comp_filter = True if comp_filter_raw is None else bool(comp_filter_raw)
        for dep in deps:
            dep_idx = STAT_INDEX[dep]
            if not explicit_filter_set[dep_idx]:
                should_filter[dep_idx] = comp_filter

        comp_formulas.append(formula_tag)
        comp_dep_names.append(tuple(deps))
        comp_min_vals.append(stat_min if has_min else 0)
        comp_max_vals.append(stat_max if has_max else 0)
        comp_has_min_vals.append(has_min)
        comp_has_max_vals.append(has_max)
        comp_weight_vals.append(stat_weight if has_weight else 0.0)

    # ------------------------------------------------------------
    # Build projected stat space (for search phase)
    # ------------------------------------------------------------
    active_indices = np.nonzero(active_mask)[0].astype(np.int32)
    proj_stats_idx = np.full(STAT_COUNT, -1, dtype=np.int32)
    proj_stats_idx[active_indices] = np.arange(len(active_indices), dtype=np.int32)

    stat_count = len(active_indices)

    min_proj = min_stats[active_indices]
    max_proj = max_stats[active_indices]
    weights_proj = weights[active_indices]

    has_min_mask_proj = has_min_mask[active_indices]
    has_max_mask_proj = has_max_mask[active_indices]

    base_min_stats_proj = base_min_stats[active_indices]
    base_max_stats_proj = base_max_stats[active_indices]

    pos_weight_mask_proj = weights_proj > 0.0
    neg_weight_mask_proj = weights_proj < 0.0

    stat_index_keys_proj = [
        next(name for name, i in STAT_INDEX.items() if i == idx)
        for idx in active_indices
    ]

    req_mask_full = np.zeros(STAT_COUNT, dtype=np.bool_)
    for name in REQ_STATS:
        req_mask_full[STAT_INDEX[name]] = True

    filter_mask_full = should_filter

    req_mask_proj = req_mask_full[active_indices]

    # META_5 (~5M rows) and META_4 (~600k rows) are dominated by branch-and-bound
    # pruning during search; the Pareto cull on those tiers costs more wall-clock
    # than it saves (measured on Armouring meteor: META_4 cull = 32s, search delta
    # = +4s, net -28s when skipped). Cap at 3 by default so cull only runs on
    # META_3 and below where it remains net-positive. Bump higher only if a
    # specific query shows BB pruning struggling.
    suggested_max_cull = 3

    req_idx = np.full(5, -1)
    i = 0
    for j, stat_idx in enumerate(active_indices):
        if stat_idx in REQ_STATS_IDX:
            req_idx[i] = j
            i += 1

    round_offset_proj = (req_mask_proj.astype(np.int32) * 50).astype(np.int32)

    # SP-cap-aware scoring metadata. For each active SP stat, look up the
    # ctx base (from `_context`), the corresponding xReq stat's projected
    # idx (or -1 if no req active), and the user's req base. The cap calc
    # at scoring time uses `max(user_req_base, current[req_idx])` for the
    # req, then clamps the bonus to [-(ctx+req), SKP_MAX-(ctx+req)].
    sp_score_active_proj = np.zeros(stat_count, dtype=np.bool_)
    sp_score_ctx_proj = np.zeros(stat_count, dtype=np.int32)
    sp_score_req_idx_proj = np.full(stat_count, -1, dtype=np.int32)
    sp_score_req_base_proj = np.zeros(stat_count, dtype=np.int32)
    _SKP_NAME_TO_CTX = {
        "str": BUILD_CTX_BASE_STR, "dex": BUILD_CTX_BASE_DEX,
        "int": BUILD_CTX_BASE_INT, "def": BUILD_CTX_BASE_DEF,
        "agi": BUILD_CTX_BASE_AGI,
    }
    _SP_TO_REQ_NAME = {"str": "strReq", "dex": "dexReq", "int": "intReq",
                       "def": "defReq", "agi": "agiReq"}
    for j, name in enumerate(stat_index_keys_proj):
        ctx_slot = _SKP_NAME_TO_CTX.get(name)
        if ctx_slot is None:
            continue
        sp_score_active_proj[j] = True
        sp_score_ctx_proj[j] = int(build_ctx_arr[ctx_slot])  # 0..SKP_MAX
        req_name = _SP_TO_REQ_NAME[name]
        req_full_idx = STAT_INDEX[req_name]
        req_proj = int(proj_stats_idx[req_full_idx])
        if req_proj >= 0:
            sp_score_req_idx_proj[j] = req_proj
        # User-provided base on the corresponding xReq (captured separately
        # in pass 1 when we encountered that req stat).
        sp_score_req_base_proj[j] = int(req_user_base_full[req_full_idx])

    # Resolve dura / duration in projected space (at most one is active).
    dura_proj_idx = -1
    for j, name in enumerate(stat_index_keys_proj):
        if name == "durability" or name == "duration":
            dura_proj_idx = j
            break

    # ------------------------------------------------------------
    # Project composite dep indices to active stat space — flat layout
    # ------------------------------------------------------------
    comp_count = len(comp_formulas)
    comp_formula_arr = np.asarray(comp_formulas, dtype=np.int32)

    # Build flat dep_indices + per-composite (offset, count) arrays.
    flat_indices = []
    offsets = []
    counts = []
    for deps in comp_dep_names:
        offsets.append(len(flat_indices))
        counts.append(len(deps))
        for name in deps:
            flat_indices.append(proj_stats_idx[STAT_INDEX[name]])

    comp_dep_offset = np.asarray(offsets, dtype=np.int32)
    comp_dep_count = np.asarray(counts, dtype=np.int32)
    comp_dep_indices = np.asarray(flat_indices, dtype=np.int32)
    comp_min_arr = np.asarray(comp_min_vals, dtype=np.int32)
    comp_max_arr = np.asarray(comp_max_vals, dtype=np.int32)
    comp_has_min_arr = np.asarray(comp_has_min_vals, dtype=np.bool_)
    comp_has_max_arr = np.asarray(comp_has_max_vals, dtype=np.bool_)
    comp_weight_arr = np.asarray(comp_weight_vals, dtype=np.float32)

    # Numba-friendly: always pass a 1D contiguous array, even if empty.
    if comp_count == 0:
        comp_formula_arr = np.zeros(0, dtype=np.int32)
        comp_dep_offset = np.zeros(0, dtype=np.int32)
        comp_dep_count = np.zeros(0, dtype=np.int32)
        comp_dep_indices = np.zeros(0, dtype=np.int32)
        comp_min_arr = np.zeros(0, dtype=np.int32)
        comp_max_arr = np.zeros(0, dtype=np.int32)
        comp_has_min_arr = np.zeros(0, dtype=np.bool_)
        comp_has_max_arr = np.zeros(0, dtype=np.bool_)
        comp_weight_arr = np.zeros(0, dtype=np.float32)

    return Query(
        search_for_inversion=search_for_inversion,
        item_type=item_type,
        skill=skill,
        min_stats=min_stats,
        max_stats=max_stats,
        weights=weights,
        has_min_mask=has_min_mask,
        has_max_mask=has_max_mask,
        base_min_stats_proj=base_min_stats_proj,
        base_max_stats_proj=base_max_stats_proj,
        active_indices=active_indices,
        stat_count=stat_count,
        min_proj=min_proj,
        max_proj=max_proj,
        weights_proj=weights_proj,
        has_min_mask_proj=has_min_mask_proj,
        has_max_mask_proj=has_max_mask_proj,
        pos_weight_mask_proj=pos_weight_mask_proj,
        neg_weight_mask_proj=neg_weight_mask_proj,
        stat_index_keys_proj=stat_index_keys_proj,
        req_mask_proj=req_mask_proj,
        req_idx=req_idx,
        filter_mask=filter_mask_full,
        proj_stats_idx=proj_stats_idx,
        round_offset_proj=round_offset_proj,
        consumable=consumable,
        suggested_max_cull=suggested_max_cull,
        dura_proj_idx=dura_proj_idx,
        build_ctx=build_ctx_arr,
        sp_score_active_proj=sp_score_active_proj,
        sp_score_ctx_proj=sp_score_ctx_proj,
        sp_score_req_idx_proj=sp_score_req_idx_proj,
        sp_score_req_base_proj=sp_score_req_base_proj,
        comp_count=comp_count,
        comp_formula=comp_formula_arr,
        comp_dep_offset=comp_dep_offset,
        comp_dep_count=comp_dep_count,
        comp_dep_indices=comp_dep_indices,
        comp_min=comp_min_arr,
        comp_max=comp_max_arr,
        comp_has_min=comp_has_min_arr,
        comp_has_max=comp_has_max_arr,
        comp_weight=comp_weight_arr,
    )


# ------------------------------------------------------------
# Projection helper
# ------------------------------------------------------------

def project_stat_matrix(stat_matrix: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
    """
    Project ingredient stat matrix into active stat space.

    Input:
        stat_matrix: [N, STAT_COUNT]

    Output:
        [N, active_stat_count]
    """
    return stat_matrix[:, active_indices]
