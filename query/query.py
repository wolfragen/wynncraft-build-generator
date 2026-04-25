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
)


_FORMULA_TAGS = {
    "mul_div_100": FORMULA_MUL_DIV_100,
    "raw_to_pct": FORMULA_RAW_TO_PCT,
    "ehp": FORMULA_EHP,
    "ehpr": FORMULA_EHPR,
}

# Arity per formula tag — used by the parser to validate dep counts.
_FORMULA_ARITY = {
    FORMULA_MUL_DIV_100: 2,
    FORMULA_RAW_TO_PCT: 2,
    FORMULA_EHP: 3,
    FORMULA_EHPR: 4,
}

# How many dep slots are stored on Query. Composites use up to MAX_COMP_ARITY
# slots; unused slots are filled with -1 sentinels.
MAX_COMP_ARITY = 4


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

    # Composite (derived) stats — Struct-of-Arrays for numba passage.
    # All indexed 0..comp_count-1. Dep indices are in projected stat space.
    # Up to MAX_COMP_ARITY (4) deps; unused slots filled with -1.
    #   arity 2 (mul_div_100, raw_to_pct):  uses dep_a, dep_b
    #   arity 3 (ehp):                       uses dep_a, dep_b, dep_c
    #   arity 4 (ehpr):                      uses all four
    comp_count: int
    comp_formula: np.ndarray        # int32[N]
    comp_dep_a_proj: np.ndarray     # int32[N]
    comp_dep_b_proj: np.ndarray     # int32[N]
    comp_dep_c_proj: np.ndarray     # int32[N]
    comp_dep_d_proj: np.ndarray     # int32[N]
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
    # Pass 1: non-composite (base) stats
    # ------------------------------------------------------------
    composite_entries = []

    for stat_name, config in user_json.items():

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

        # Pad deps tuple to MAX_COMP_ARITY with None.
        padded = list(deps) + [None] * (MAX_COMP_ARITY - len(deps))

        comp_formulas.append(formula_tag)
        comp_dep_names.append(tuple(padded))
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
    # Project composite dep indices to active stat space
    # ------------------------------------------------------------
    comp_count = len(comp_formulas)
    comp_formula_arr = np.asarray(comp_formulas, dtype=np.int32)

    # Build per-slot projected dep arrays. Unused slots (None) → -1 sentinel.
    def _proj_or_sentinel(name):
        if name is None:
            return np.int32(-1)
        return np.int32(proj_stats_idx[STAT_INDEX[name]])

    comp_dep_a_proj = np.array(
        [_proj_or_sentinel(deps[0]) for deps in comp_dep_names], dtype=np.int32,
    )
    comp_dep_b_proj = np.array(
        [_proj_or_sentinel(deps[1]) for deps in comp_dep_names], dtype=np.int32,
    )
    comp_dep_c_proj = np.array(
        [_proj_or_sentinel(deps[2]) for deps in comp_dep_names], dtype=np.int32,
    )
    comp_dep_d_proj = np.array(
        [_proj_or_sentinel(deps[3]) for deps in comp_dep_names], dtype=np.int32,
    )
    comp_min_arr = np.asarray(comp_min_vals, dtype=np.int32)
    comp_max_arr = np.asarray(comp_max_vals, dtype=np.int32)
    comp_has_min_arr = np.asarray(comp_has_min_vals, dtype=np.bool_)
    comp_has_max_arr = np.asarray(comp_has_max_vals, dtype=np.bool_)
    comp_weight_arr = np.asarray(comp_weight_vals, dtype=np.float32)

    # Numba-friendly: always pass a 1D contiguous array, even if empty.
    if comp_count == 0:
        comp_formula_arr = np.zeros(0, dtype=np.int32)
        comp_dep_a_proj = np.zeros(0, dtype=np.int32)
        comp_dep_b_proj = np.zeros(0, dtype=np.int32)
        comp_dep_c_proj = np.zeros(0, dtype=np.int32)
        comp_dep_d_proj = np.zeros(0, dtype=np.int32)
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
        comp_count=comp_count,
        comp_formula=comp_formula_arr,
        comp_dep_a_proj=comp_dep_a_proj,
        comp_dep_b_proj=comp_dep_b_proj,
        comp_dep_c_proj=comp_dep_c_proj,
        comp_dep_d_proj=comp_dep_d_proj,
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
