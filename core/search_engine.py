"""
search_engine.py

Full craft search engine.

Searches across all meta-set groups (n = 0..5),
evaluates crafts directly, and returns the best solution.

Returns:
    dict with:
        - meta
        - full_slots (int32[6])
    or None if no valid craft.
"""

import numpy as np
from core.craft_state import CraftState
from core.leaf_evaluator import evaluate_leaf

from time import time


def search(all_meta_sets, db, query):

    best_score = -np.inf
    best_meta = None
    best_full_slots = None

    # ------------------------------------------------------------
    # Required coverage mask
    # Rule:
    #   stat must appear if:
    #       min > 0 OR max < 0
    # ------------------------------------------------------------
    required_mask = 0

    for s in range(query.stat_count):
        if query.min_proj[s] > 0 or query.max_proj[s] < 0:
            required_mask |= (1 << s)
            
    durability_idx = None

    for i, name in enumerate(query.stat_index_keys_proj):
        if name == "durability":
            durability_idx = i
            break
        
    if(durability_idx is None):
        print("minimum durability wasn't found in the query")
        return

    # ------------------------------------------------------------
    # Iterate meta groups (n = 0..5)
    # ------------------------------------------------------------
    for n_meta, meta_group in enumerate(all_meta_sets):
        start_time = time()
        k = 6-n_meta

        if not meta_group:
            continue

        avoid_permutations = (n_meta == 0)
        
        for meta in meta_group:
    
            state = CraftState(k)
    
            # -------------------------
            # Compute initial coverage from meta
            # -------------------------
            meta_coverage_mask = 0
    
            for s in range(query.stat_count):
                min_v = meta["base_min_proj"][s]
                max_v = meta["base_max_proj"][s]
    
                if min_v != 0 or max_v != 0:
                    meta_coverage_mask |= (1 << s)
    
            # -------------------------
            # DFS
            # -------------------------
            def dfs(start_idx, coverage_mask):
    
                nonlocal best_score, best_meta, best_full_slots
    
                depth = state.depth
    
                if depth == k:
    
                    if (coverage_mask & required_mask) == required_mask:
    
                        score, min_vals, max_vals = evaluate_leaf(
                            state.ingredients,
                            k,
                            db.stat_min_matrix,
                            db.stat_max_matrix,
                            meta["base_min_proj"],
                            meta["base_max_proj"],
                            meta["void_effectiveness"],
                            use_eff=(n_meta != 0),
                            durability_idx=durability_idx,
                            has_min_mask=query.has_min_mask_proj,
                            has_max_mask=query.has_max_mask_proj,
                            min_vals=query.min_proj,
                            max_vals=query.max_proj,
                            weights=query.weights_proj,
                        )
    
                        if score > best_score:
                            best_score = score
                            best_meta = meta
    
                            full_slots = meta["ings"].copy()
                            for i, slot in enumerate(meta["void_positions"]):
                                db_idx = state.ingredients[i]
                                full_slots[slot] = int(db.json_ids[db_idx])
    
                            best_full_slots = full_slots.copy()
    
                    return
    
                remaining = k - depth
                uncovered = required_mask & ~coverage_mask
    
                if uncovered.bit_count() > remaining:
                    return
    
                loop_start = start_idx if avoid_permutations else 0
    
                for i in range(loop_start, db.count):
    
                    state.apply(i)
    
                    new_mask = coverage_mask | db.stat_bitmask[i]
    
                    if avoid_permutations:
                        dfs(i, new_mask)
                    else:
                        dfs(0, new_mask)
    
                    state.undo()
    
            dfs(0, meta_coverage_mask)
        print(f"Finished the {n_meta}-meta_sets in {time() - start_time:.0f}s")

    if best_meta is None:
        return None

    return best_full_slots










