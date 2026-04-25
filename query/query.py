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
from data.skillpoint_lookup import (
    SKP_ELEMENT_PCT,
    SKP_STR, SKP_DEX, SKP_INT, SKP_DEF, SKP_AGI, SKP_MAX,
)


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
# Values come from SKP_ELEMENT_PCT (skillpoint_damage_mult variant), which is
# what wynnbuilder's calculateSpellDamage uses for both per-element boosts AND
# the global str_boost. For str/dex/def/agi these are identical to the
# headline values; for int the damage version uses 1.0 mult (vs 0.619 for
# mana cost reduction, which we don't currently compute).
BUILD_CTX_STR_PCT = 0     # skillPointsToPercentage(str) * 1.0
BUILD_CTX_DEX_PCT = 1     # skillPointsToPercentage(dex) * 1.0
BUILD_CTX_INT_PCT = 2     # skillPointsToPercentage(int) * 1.0 (water dmg boost)
BUILD_CTX_DEF_PCT = 3     # ... * 0.867
BUILD_CTX_AGI_PCT = 4     # ... * 0.951
BUILD_CTX_ATK_SPD = 5     # baseDamageMultiplier[atk_speed]
BUILD_CTX_CRIT_PCT = 6    # crit damage % (default 0)
BUILD_CTX_SIZE = 7


def _build_ctx_array(ctx_dict):
    """
    Build the float64 build-context array consumed by spell evaluators.
    `ctx_dict` is the user's `_context` entry, e.g.
        {"str": 80, "dex": 0, "int": 100, "def": 0, "agi": 40,
         "atk_speed": "NORMAL", "crit_dam_pct": 0}
    All keys optional; defaults: skillpoints=0, atk_speed="NORMAL", crit=0.
    """
    arr = np.zeros(BUILD_CTX_SIZE, dtype=np.float64)
    if ctx_dict:
        for skp_name, skp_idx in (("str", SKP_STR), ("dex", SKP_DEX),
                                   ("int", SKP_INT), ("def", SKP_DEF),
                                   ("agi", SKP_AGI)):
            count = int(ctx_dict.get(skp_name, 0))
            if count < 0:
                count = 0
            elif count > SKP_MAX:
                count = SKP_MAX
            arr[skp_idx] = float(SKP_ELEMENT_PCT[skp_idx, count])
        atk_speed = ctx_dict.get("atk_speed", "NORMAL")
        arr[BUILD_CTX_ATK_SPD] = float(ATK_SPEED_MULT.get(atk_speed, ATK_SPEED_MULT["NORMAL"]))
        arr[BUILD_CTX_CRIT_PCT] = float(ctx_dict.get("crit_dam_pct", 0))
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

    consumable: bool
    suggested_max_cull: int

    # Projected-space index of durability (crafted) or duration (consumable).
    # -1 if neither is active — search will abort in that case.
    dura_proj_idx: int

    # Build-context constants (skillpoint percentages, atk_speed mult, crit pct)
    # passed verbatim to numba kernels. See BUILD_CTX_* indices above.
    build_ctx: np.ndarray  # float64[BUILD_CTX_SIZE]

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

    suggested_max_cull = 5  # For meta-sets culling.
    if any(req for req in req_mask_proj):
        suggested_max_cull = 4  # Si on a au moins un req défini, il vaut mieux ne pas cull le 5

    req_idx = np.full(5, -1)
    i = 0
    for j, stat_idx in enumerate(active_indices):
        if stat_idx in REQ_STATS_IDX:
            req_idx[i] = j
            i += 1

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
        consumable=consumable,
        suggested_max_cull=suggested_max_cull,
        dura_proj_idx=dura_proj_idx,
        build_ctx=build_ctx_arr,
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
