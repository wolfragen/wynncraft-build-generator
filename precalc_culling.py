import json
import os
import sys
import numpy as np
from numba import njit

REQ_STATS = {"strReq", "dexReq", "intReq", "defReq", "agiReq"}

# ---------------------------------------------------------
# Numba JIT Compiled Functions for blazing fast comparisons
# ---------------------------------------------------------
@njit(fastmath=True)
def compare_vectors(a, b):
    """Returns 1 if a > b, -1 if b > a, 2 if a == b, 0 if incomparable"""
    a_better = False
    b_better = False
    for i in range(len(a)):
        if a[i] > b[i]:
            a_better = True
        elif b[i] > a[i]:
            b_better = True
            
        # Early exit if neither strictly dominates
        if a_better and b_better:
            return 0 
            
    if a_better: return 1
    if b_better: return -1
    return 2 # Identical

@njit
def pareto_filter(matrix):
    """Computes the Pareto frontier over a 2D numpy array."""
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)
    
    for i in range(n):
        if not is_kept[i]: 
            continue
            
        for j in range(i + 1, n):
            if not is_kept[j]: 
                continue
            
            cmp = compare_vectors(matrix[i], matrix[j])
            
            if cmp == 1:
                # i dominates j
                is_kept[j] = False
            elif cmp == 2:
                # Identical, kill the duplicate
                is_kept[j] = False
            elif cmp == -1:
                # j dominates i
                is_kept[i] = False
                break # i is dead, move to next i
                
    return is_kept

# ---------------------------------------------------------
# Main Python Wrapper
# ---------------------------------------------------------
def cull_recipes(filename):
    in_path = os.path.join("data", "precalc/full", filename)
    out_path = os.path.join("data", "precalc", "generic_cull", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    print("Pass 1/3: Reading file and mapping stat keys...")
    stat_keys = set()
    raw_lines = []
    
    # Accept user's varied keys (ingredients/ings, effectiveness/eff)
    num_effs = 0 

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().rstrip(',')
            if clean in ('[', ']', ''): continue
            
            raw_lines.append(clean)
            
            try:
                cand = json.loads(clean)
                if "stats" in cand:
                    stat_keys.update(cand["stats"].keys())
                
                # Determine how many "-1" efficiency slots there are (only need to do once)
                if num_effs == 0:
                    ings_list = cand.get("ingredients", cand.get("ings", []))
                    eff_list = cand.get("effectiveness", cand.get("eff", []))
                    num_effs = len([e for i, e in zip(ings_list, eff_list) if i == -1])
            except:
                continue

    if not raw_lines:
        print("No valid recipes found.")
        return

    stat_list = sorted(list(stat_keys))
    stat_to_idx = {k: i for i, k in enumerate(stat_list)}
    num_stats = len(stat_list)
    num_lines = len(raw_lines)
    
    print(f"Pass 2/3: Building {num_lines}x{num_effs + num_stats} numerical matrix...")
    matrix = np.zeros((num_lines, num_effs + num_stats), dtype=np.float32)
    
    for row_idx, clean in enumerate(raw_lines):
        cand = json.loads(clean)
        ings_list = cand.get("ingredients", cand.get("ings", []))
        eff_list = cand.get("effectiveness", cand.get("eff", []))
        
        # 1. Process Efficiencies for -1 slots
        effs = []
        for ing_id, e in zip(ings_list, eff_list):
            if ing_id == -1:
                # Strip '%' if it is a string
                effs.append(float(str(e).replace('%', '')))
                
        effs.sort(reverse=True)
        for i, val in enumerate(effs):
            matrix[row_idx, i] = val
            
        # 2. Process Stats
        if "stats" in cand:
            for stat_name, d_val in cand["stats"].items():
                col_idx = num_effs + stat_to_idx[stat_name]
                
                # Flip Reqs so that -50 is 'worse' than 0
                if stat_name in REQ_STATS:
                    matrix[row_idx, col_idx] = -d_val["min"]
                else:
                    matrix[row_idx, col_idx] = d_val["max"]

    print("Pass 3/3: Running Numba dominance cull...")
    # OPTIMIZATION: Sort by the sum of metrics descending. 
    # This guarantees we evaluate heavily-statted items first, skipping massive amounts of loops.
    sort_order = np.argsort(-np.sum(matrix, axis=1))
    matrix = matrix[sort_order]
    
    # Reorder raw lines to match matrix
    raw_lines = [raw_lines[i] for i in sort_order]
    
    # Execute JIT function
    is_kept = pareto_filter(matrix)
    kept_indices = np.where(is_kept)[0]
    
    print("Writing output...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, idx in enumerate(kept_indices):
            f.write(raw_lines[idx])
            f.write(",\n" if i < len(kept_indices) - 1 else "\n")
        f.write("]")
        
    print(f"\nDone! Total: {num_lines} | Kept: {len(kept_indices)}")

if __name__ == "__main__":
    cull_recipes("JEWELING_META_1.json")


