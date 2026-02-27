"""
search_engine.py
"""

import numpy as np
from numba import njit
from time import time

from core.pruning import prune


# ============================================================
# DFS (numba)
# ============================================================

@njit
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
    db_count,
    meta_void_eff,
    durability_idx,
    has_min_mask,
    has_max_mask,
    min_vals,
    max_vals,
    weights,
):
    # Leaf
    if depth == k:

        score = 0.0

        # Evaluate directly using current stats
        for s in range(len(current_min)):

            min_v = current_min[s]
            max_v = current_max[s]

            if has_min_mask[s] and max_v < min_vals[s]:
                return

            if has_max_mask[s] and min_v > max_vals[s]:
                return

            score += weights[s] * max_v * 0.99
            score += weights[s] * min_v * 0.01

        if score > best_score_ref[0]:
            best_score_ref[0] = score
            for i in range(k):
                best_solution[i] = ingredients[i]
        return

    # Pruning hook (stat-based pruning to be improved later)
    if prune(depth, k, current_min, current_max):
        return

    loop_start = start_index if k == 6 else 0

    for i in range(loop_start, db_count):

        ingredients[depth] = i

        eff = meta_void_eff[depth]

        # Apply ingredient contribution
        for s in range(len(current_min)):
            if(s == durability_idx):
                current_min[s] += db_stat_min[i, s]
                current_max[s] += db_stat_max[i, s]
            else:
                current_min[s] += (db_stat_min[i, s] * eff) //100
                current_max[s] += (db_stat_max[i, s] * eff) //100

        next_start = i if k == 6 else 0 # Indice de départ, évite les permutations
        # TODO : voir pour k = 5, k = 4 et peut-être même k = 3

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
            db_count,
            meta_void_eff,
            durability_idx,
            has_min_mask,
            has_max_mask,
            min_vals,
            max_vals,
            weights,
        )

        # Undo
        for s in range(len(current_min)):
            if(s == durability_idx):
                current_min[s] -= db_stat_min[i, s]
                current_max[s] -= db_stat_max[i, s]
            else:
                current_min[s] -= (db_stat_min[i, s] * eff) //100
                current_max[s] -= (db_stat_max[i, s] * eff) //100


# ============================================================
# Search One Meta Batch (numba)
# ============================================================

@njit
def search_meta_batch(
    ings_matrix,
    void_count,
    void_eff_matrix,
    base_min_matrix,
    base_max_matrix,
    db_stat_min,
    db_stat_max,
    db_count,
    durability_idx,
    has_min_mask,
    has_max_mask,
    min_vals,
    max_vals,
    weights,
):
    M = ings_matrix.shape[0]
    k = void_count

    best_score = -1e18
    best_meta_index = -1

    ingredients = np.zeros(k, dtype=np.int32)
    best_solution = np.zeros(k, dtype=np.int32)
    best_score_ref = np.zeros(1, dtype=np.float64)

    for m in range(M):

        current_min = base_min_matrix[m].copy()
        current_max = base_max_matrix[m].copy()

        best_score_ref[0] = -1e18

        dfs(
            0,
            0,
            k,
            ingredients,
            current_min,
            current_max,
            best_score_ref,
            best_solution,
            db_stat_min,
            db_stat_max,
            db_count,
            void_eff_matrix[m],
            durability_idx,
            has_min_mask,
            has_max_mask,
            min_vals,
            max_vals,
            weights,
        )

        if best_score_ref[0] > best_score:
            best_score = best_score_ref[0]
            best_meta_index = m

    return best_score, best_meta_index, best_solution


# ============================================================
# Python Orchestration
# ============================================================

def search(all_meta_sets, db, query):

    best_score = -np.inf
    best_full_slots = None

    durability_idx = -1
    for i, name in enumerate(query.stat_index_keys_proj):
        if name == "durability":
            durability_idx = i
            break
        

    for meta_batch in all_meta_sets:
        start_time = time()

        if meta_batch.ings_matrix.shape[0] == 0:
            continue

        score, meta_index, sol = search_meta_batch(
            meta_batch.ings_matrix,
            meta_batch.void_count,
            meta_batch.void_eff_matrix,
            meta_batch.base_min_matrix,
            meta_batch.base_max_matrix,
            db.stat_min_matrix,
            db.stat_max_matrix,
            db.count,
            durability_idx,
            query.has_min_mask_proj,
            query.has_max_mask_proj,
            query.min_proj,
            query.max_proj,
            query.weights_proj,
        )

        if score > best_score and meta_index != -1:

            best_score = score

            meta_ings = meta_batch.ings_matrix[meta_index]
            full_slots = meta_ings.copy()

            idx = 0
            for slot in range(6):
                if full_slots[slot] == -1:
                    db_idx = sol[idx]
                    full_slots[slot] = db.json_ids[db_idx]
                    idx += 1

            best_full_slots = full_slots
            
        print(f"meta batch {6-meta_batch.void_count}: {len(meta_batch.ings_matrix)}, time elapsed: {time()-start_time:.0f}s")
        print(f"Best: score={score:.0f} ")

    return best_full_slots