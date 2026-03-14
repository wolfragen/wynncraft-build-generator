"""
meta_set_loader.py

Loads and refines meta-sets according to Query.

Structure based on META JSON:
- "ings": length-6 list
- "eff": length-6 list (percentage)
- "stats": dict of min/max values
"""

import numpy as np
from numba import njit
import json
import os
from typing import NamedTuple

from data.stats import STAT_INDEX, STAT_COUNT, REQ_STATS


class MetaBatch(NamedTuple):
    ings_matrix: np.ndarray          # (M, 6)
    void_count: int                  # number of void slots
    void_eff_matrix: np.ndarray      # (M, void_count)
    void_slots_matrix: np.ndarray    # (M, void_count)
    base_min_matrix: np.ndarray      # (M, K)
    base_max_matrix: np.ndarray      # (M, K)


# ------------------------------------------------------------
# Main Loader
# ------------------------------------------------------------

def load_meta_sets(skill, query, recipe, culling=True, should_print=False, base_path="data/precalc/generic_cull"):

    full_meta_sets = []

    # META_0
    first_raw = [{
        "ings": [-1, -1, -1, -1, -1, -1],
        "eff": [100, 100, 100, 100, 100, 100],
        "stats": {}
    }]

    first_batch = refine_meta_sets(first_raw, query, recipe, culling=False)
    full_meta_sets.append(first_batch)

    # META_1..5
    for n in range(1, 6):

        raw_meta_sets = load_raw_meta_sets(skill, n)

        batch = refine_meta_sets(
            raw_meta_sets,
            query,
            recipe,
            culling=culling
        )

        if should_print:
            print(f"{skill}_META_{n}: {len(raw_meta_sets)} => {batch.ings_matrix.shape[0]}")

        full_meta_sets.append(batch)

    return full_meta_sets


def load_raw_meta_sets(skill: str, n: int, base_path="data/precalc/generic_cull"):

    filename = f"{skill.upper()}_META_{n}.json"
    path = os.path.join(base_path, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ------------------------------------------------------------
# Refine + Optional Cull
# ------------------------------------------------------------

def refine_meta_sets(raw_meta_sets: list, query, recipe, culling=True):

    if not isinstance(raw_meta_sets, list):
        raise TypeError("refine_meta_sets expects a list of meta-set dicts.")

    num_sets = len(raw_meta_sets)
    active_indices = query.active_indices
    num_stats = query.stat_count

    if num_sets == 0:
        return MetaBatch(
            np.empty((0, 6), dtype=np.int32),
            0,
            np.empty((0, 6), dtype=np.int32),
            np.empty((0, 6), dtype=np.int32),
            np.empty((0, num_stats), dtype=np.int32),
            np.empty((0, num_stats), dtype=np.int32),
        )

    first_ings = raw_meta_sets[0]["ings"]
    void_count = sum(1 for v in first_ings if v == -1)

    ings_matrix = np.zeros((num_sets, 6), dtype=np.int32)
    void_eff_matrix = np.zeros((num_sets, void_count), dtype=np.int32)
    void_real_slot_matrix = np.zeros((num_sets, void_count), dtype=np.int32)

    base_min_matrix = np.zeros((num_sets, num_stats), dtype=np.int32)
    base_max_matrix = np.zeros((num_sets, num_stats), dtype=np.int32)

    for i, raw_meta_set in enumerate(raw_meta_sets):

        ings = raw_meta_set["ings"]
        ings_matrix[i] = ings

        temp_count = 0
        full_eff = raw_meta_set["eff"]
        for slot in range(6):
            if ings[slot] == -1:
                void_eff_matrix[i, temp_count] = full_eff[slot]
                void_real_slot_matrix[i, temp_count] = slot
                temp_count += 1

        base_min_full = np.zeros(STAT_COUNT, dtype=np.int32)
        base_max_full = np.zeros(STAT_COUNT, dtype=np.int32)

        raw_stats = raw_meta_set.get("stats", {})

        for stat_name, value in raw_stats.items():

            idx = STAT_INDEX.get(stat_name)
            if idx is None:
                continue

            min_val = value.get("min", value.get("minimum", 0))
            max_val = value.get("max", value.get("maximum", 0))

            base_min_full[idx] = min_val
            base_max_full[idx] = max_val

        base_min_proj = base_min_full[active_indices] + recipe.base_min_stats_proj
        base_max_proj = base_max_full[active_indices] + recipe.base_max_stats_proj

        base_min_matrix[i] = base_min_proj
        base_max_matrix[i] = base_max_proj

    if culling and num_sets > 0:

        req_mask_full = np.zeros(STAT_COUNT, dtype=np.bool_)
        for name in REQ_STATS:
            req_mask_full[STAT_INDEX[name]] = True

        req_mask_proj = req_mask_full[active_indices]

        kept_indices = numba_cull(
            base_min_matrix,
            base_max_matrix,
            void_eff_matrix,
            ings_matrix,
            void_count,
            req_mask_proj
        )

        ings_matrix = ings_matrix[kept_indices]
        void_eff_matrix = void_eff_matrix[kept_indices]
        void_real_slot_matrix = void_real_slot_matrix[kept_indices]
        base_min_matrix = base_min_matrix[kept_indices]
        base_max_matrix = base_max_matrix[kept_indices]

    M = void_eff_matrix.shape[0]

    sorted_void_eff_matrix = np.zeros_like(void_eff_matrix)
    void_slots_matrix = np.zeros_like(void_real_slot_matrix)

    for m in range(M):

        order = np.argsort(void_eff_matrix[m])

        for j in range(void_count):
            src = order[void_count - 1 - j]
            sorted_void_eff_matrix[m, j] = void_eff_matrix[m, src]
            void_slots_matrix[m, j] = void_real_slot_matrix[m, src]

    return MetaBatch(
        ings_matrix=ings_matrix,
        void_count=void_count,
        void_eff_matrix=sorted_void_eff_matrix,
        void_slots_matrix=void_slots_matrix,
        base_min_matrix=base_min_matrix,
        base_max_matrix=base_max_matrix,
    )


# ------------------------------------------------------------
# Pareto
# ------------------------------------------------------------

@njit(fastmath=True)
def compare_vectors(a, b, num_effs):
    """
    Returns:
       1 if A dominates B
      -1 if B dominates A
       2 if A and B are identical
       0 if A and B are incomparable
    """
    a_better = False
    b_better = False
    
    for i in range(len(a)):
        va = a[i]
        vb = b[i]
        
        if va == vb:
            continue
            
        if i < num_effs:
            # --- CUSTOM EFFECTIVENESS LOGIC ---
            # 1. Zero is strictly worse than any non-zero
            if va == 0:
                b_better = True
            elif vb == 0:
                a_better = True
            # 2. Positive and Negative are incomparable
            elif (va > 0 and vb < 0) or (va < 0 and vb > 0):
                return 0 
            # 3. If both positive: higher is better
            elif va > 0: 
                if va > vb: a_better = True
                else: b_better = True
            # 4. If both negative: lower (further from zero) is better
            else: 
                if va < vb: a_better = True # e.g., -100 < -50
                else: b_better = True
        else:
            # --- STANDARD STAT LOGIC ---
            if va > vb: a_better = True
            else: b_better = True
            
        # Early exit: if both have a "better" dimension, they are incomparable
        if a_better and b_better:
            return 0 
            
    if a_better: return 1
    if b_better: return -1
    return 2 # Identical


@njit
def pareto_filter(matrix, num_effs):
    """Computes the Pareto frontier using the modified effectiveness rules."""
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)
    
    for i in range(n):
        if not is_kept[i]: 
            continue
            
        for j in range(i + 1, n):
            if not is_kept[j]: 
                continue
            
            cmp = compare_vectors(matrix[i], matrix[j], num_effs)
            
            if cmp == 1:
                is_kept[j] = False
            elif cmp == 2:
                is_kept[j] = False
            elif cmp == -1:
                is_kept[i] = False
                break 
                
    return is_kept


@njit
def numba_cull(base_min_matrix,
               base_max_matrix,
               void_eff_matrix,
               ings_matrix,
               void_count,
               req_mask_proj):

    num_sets, num_stats = base_min_matrix.shape
    matrix = np.zeros((num_sets, void_count + num_stats), dtype=np.int32)

    for i in range(num_sets):

        # --------------------------
        # Sort descending
        # --------------------------
        tmp_eff = void_eff_matrix[i].copy()
        tmp_eff.sort()  # ascending

        for j in range(void_count):
            matrix[i, j] = tmp_eff[void_count - 1 - j]

        # --------------------------
        # Stats
        # --------------------------
        for s in range(num_stats):
            if req_mask_proj[s]:
                matrix[i, void_count + s] = -base_min_matrix[i, s]
            else:
                matrix[i, void_count + s] = base_max_matrix[i, s]

    is_kept = pareto_filter(matrix, void_count)
    return np.where(is_kept)[0]