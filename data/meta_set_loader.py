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

from data.stats import STAT_INDEX, STAT_COUNT, REQ_STATS


def load_meta_sets(skill, query, recipe, culling=True, should_print=False, base_path="data/precalc/generic_cull"):
    full_meta_sets = []
    
    first_meta_set = refine_meta_sets([{
        "ings":[-1,-1,-1,-1,-1,-1],
        "eff":[100,100,100,100,100,100],
        "stats": {}}], query, recipe)
    full_meta_sets.append(first_meta_set)
    
    for n in range(1,6):
        raw_meta_sets = load_raw_meta_sets(skill, n)
        refined_meta_sets = refine_meta_sets(raw_meta_sets, query, recipe)
        
        if(culling):
            meta_sets = cull_refined_meta_sets(refined_meta_sets, query)
            if(should_print):
                print(f"{skill}_META_{n}: {len(refined_meta_sets)} => {len(meta_sets)}, {len(meta_sets)/len(refined_meta_sets)*100:.1f}% restants")
            full_meta_sets.append(meta_sets)
            
        else:
            full_meta_sets.append(refined_meta_sets)
    return full_meta_sets

def print_meta_sets(meta_sets, query):

    for i, meta in enumerate(meta_sets):

        print(f"\n===== META SET {i} =====")

        # -----------------------------
        # Meta ingredient IDs
        # -----------------------------
        print("ings :", meta["ings"])

        # -----------------------------
        # Void info
        # -----------------------------
        print("void_effectiveness :", meta["void_effectiveness"])

        # -----------------------------
        # Stats
        # -----------------------------
        for idx, stat_name in enumerate(query.stat_index_keys_proj):

            min_val = meta["base_min_proj"][idx]
            max_val = meta["base_max_proj"][idx]

            print(f"{stat_name} : {min_val}/{max_val}")


def load_raw_meta_sets(skill: str, n: int, base_path="data/precalc/generic_cull"):
    """
    Load raw meta-sets JSON file.

    Args:
        skill: crafting profession (e.g. "ARMOURING")
        n: number of meta-ingredients
        base_path: folder containing META files

    Returns:
        list of raw meta-set dicts
    """

    filename = f"{skill.upper()}_META_{n}.json"
    path = os.path.join(base_path, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ------------------------------------------------------------
# Refine all meta sets
# ------------------------------------------------------------

def refine_meta_sets(raw_meta_sets: list, query, recipe):
    """
    Refine a list of raw meta-sets according to Query.

    Args:
        raw_meta_sets: list of dict (output of load_raw_meta_sets)
        query: Query instance

    Returns:
        list of refined meta-sets
    """

    if not isinstance(raw_meta_sets, list):
        raise TypeError("refine_meta_sets expects a list of meta-set dicts.")

    active_indices = query.active_indices

    refined_sets = []

    for raw_meta_set in raw_meta_sets:

        # ------------------------------------------------------------
        # Slots
        # ------------------------------------------------------------
        ings = raw_meta_set["ings"]

        void_positions = [i for i, v in enumerate(ings) if v == -1]
        void_count = len(void_positions)

        # ------------------------------------------------------------
        # Base stats (min/max full → projected)
        # ------------------------------------------------------------
        base_min_full = np.zeros(STAT_COUNT, dtype=np.int32)
        base_max_full = np.zeros(STAT_COUNT, dtype=np.int32)

        raw_stats = raw_meta_set.get("stats", {})

        for stat_name, value in raw_stats.items():

            idx = STAT_INDEX.get(stat_name)
            if idx is None:
                continue

            if isinstance(value, dict):
                min_val = value.get("min", value.get("minimum", 0))
                max_val = value.get("max", value.get("maximum", 0))
            else:
                min_val = value
                max_val = value

            base_min_full[idx] = min_val
            base_max_full[idx] = max_val

        base_min_proj = base_min_full[active_indices]
        base_max_proj = base_max_full[active_indices]
        
        # ------------------------------------------------------------
        # Inject recipe durability (if durability is active)
        # ------------------------------------------------------------
        dur_idx_full = STAT_INDEX.get("durability")
        
        if dur_idx_full is not None:
        
            # Check if durability is part of projected stats
            for proj_idx, full_idx in enumerate(query.active_indices):
        
                if full_idx == dur_idx_full:
        
                    base_min_proj[proj_idx] += recipe.scaled_dura_min
                    base_max_proj[proj_idx] += recipe.scaled_dura_max
                    break

        # ------------------------------------------------------------
        # Effectiveness
        # ------------------------------------------------------------
        full_eff_percent = raw_meta_set["eff"]  # length 6
        full_eff = np.array(full_eff_percent, dtype=np.int32)

        void_effectiveness = full_eff[void_positions]

        refined_sets.append({
            "ings": ings.copy(),
            "void_positions": void_positions,
            "void_count": void_count,
            "void_effectiveness": void_effectiveness,
            "base_min_proj": base_min_proj,
            "base_max_proj": base_max_proj,
        })

    return refined_sets



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


def cull_refined_meta_sets(refined_meta_sets, query):
    """
    Apply same Pareto dominance logic as original precalc file,
    but using projected meta-set stats.
    """

    if not refined_meta_sets:
        return refined_meta_sets

    num_sets = len(refined_meta_sets)
    num_effs = max(meta["void_count"] for meta in refined_meta_sets)
    num_stats = query.stat_count

    matrix = np.zeros((num_sets, num_effs + num_stats), dtype=np.float32)

    # Map projected stat index to stat name
    stat_names = list(query.stat_index_keys_proj)

    for i, meta in enumerate(refined_meta_sets):

        # -------------------------------
        # Effectiveness
        # -------------------------------
        effs = list(meta["void_effectiveness"])
        effs.sort(reverse=True)

        for j, val in enumerate(effs):
            matrix[i, j] = val

        # -------------------------------
        # Stats
        # -------------------------------
        for s_idx in range(num_stats):

            stat_name = stat_names[s_idx]

            if stat_name in REQ_STATS:
                matrix[i, num_effs + s_idx] = -meta["base_min_proj"][s_idx]
            else:
                matrix[i, num_effs + s_idx] = meta["base_max_proj"][s_idx]

    # Sorting optimization (same as original)
    eff_scores = np.sum(np.abs(matrix[:, :num_effs]), axis=1)
    stat_scores = np.sum(matrix[:, num_effs:], axis=1)
    sort_order = np.argsort(-(eff_scores + stat_scores))

    matrix = matrix[sort_order]
    refined_meta_sets = [refined_meta_sets[i] for i in sort_order]

    is_kept = pareto_filter(matrix, num_effs)
    kept_indices = np.where(is_kept)[0]

    return [refined_meta_sets[i] for i in kept_indices]






