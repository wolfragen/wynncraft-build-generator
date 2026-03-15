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

from data.stats import STAT_INDEX, STAT_COUNT, REQ_STATS, DERIVED_DEPENDENCIES


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
    filter_mask: np.ndarray
    proj_stats_idx: np.ndarray
    
    consumable: bool
    suggested_max_cull: int


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

    # ------------------------------------------------------------
    # Parse user JSON
    # ------------------------------------------------------------
    should_filter = np.zeros(STAT_COUNT, dtype=np.bool_)
    for stat_name, config in user_json.items():
        
        deps = DERIVED_DEPENDENCIES.get(stat_name)

        if deps is not None:
            for dep in deps:
                idx = STAT_INDEX[dep]
                active_mask[idx] = True
            
            stat_min = config.get("min")
            stat_max = config.get("max")
            stat_weight = config.get("weight")
            #TODO derived_min and max
            
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
            
        stat_base = config.get("base")
        stat_min_base = config.get("min_base")
        stat_max_base = config.get("max_base")
        
        if stat_base is not None:
            base_min_stats[idx] = stat_base
            base_max_stats[idx] = stat_base
        
        if stat_min_base is not None:
            base_min_stats[idx] = stat_min_base
        
        if stat_max_base is not None:
            base_max_stats[idx] = stat_max_base
            
        should_filter_stat = config.get("ingredient_filter")
        if should_filter_stat is not None and should_filter_stat == False:
            should_filter[idx] = False

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

    pos_weight_mask_proj = weights_proj >= 0.0
    neg_weight_mask_proj = weights_proj <= 0.0

    stat_index_keys_proj = [
        next(name for name, i in STAT_INDEX.items() if i == idx)
        for idx in active_indices
    ]
    
    req_mask_full = np.zeros(STAT_COUNT, dtype=np.bool_)
    filter_mask_full = np.zeros(STAT_COUNT, dtype=np.bool_)
    for name in REQ_STATS:
        req_mask_full[STAT_INDEX[name]] = True
        if should_filter[STAT_INDEX[name]]:
            filter_mask_full[STAT_INDEX[name]] = True
    
    req_mask_proj = req_mask_full[active_indices]
    
    suggested_max_cull = 5 # For meta-sets culling.
    if any(req for req in req_mask_proj):
        suggested_max_cull = 4 # Si on a au moins un req défini, il vaut mieux ne pas cull le 5
    

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
        req_mask_proj = req_mask_proj,
        filter_mask=filter_mask_full,
        proj_stats_idx=proj_stats_idx,
        consumable=consumable,
        suggested_max_cull=suggested_max_cull,
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