import json
import os
import sys
import numpy as np
from numba import njit

REQ_STATS = {"strReq", "dexReq", "intReq", "defReq", "agiReq"}

# ---------------------------------------------------------
# Numba JIT Compiled Functions
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Main Python Wrapper
# ---------------------------------------------------------
def cull_recipes(filename):
    in_path = os.path.join("data", "precalc", "full", filename)
    out_path = os.path.join("data", "precalc", "generic_cull", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    print("Pass 1/3: Reading and mapping stats...")
    stat_keys = set()
    raw_lines = []
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
                if num_effs == 0:
                    ings = cand.get("ingredients", cand.get("ings", []))
                    effs = cand.get("effectiveness", cand.get("eff", []))
                    num_effs = len([e for i, e in zip(ings, effs) if i == -1])
            except: continue

    if not raw_lines: return

    stat_list = sorted(list(stat_keys))
    stat_to_idx = {k: i for i, k in enumerate(stat_list)}
    num_stats = len(stat_list)
    num_lines = len(raw_lines)
    
    print(f"Pass 2/3: Building {num_lines}x{num_effs + num_stats} matrix...")
    matrix = np.zeros((num_lines, num_effs + num_stats), dtype=np.float32)
    
    for row_idx, clean in enumerate(raw_lines):
        cand = json.loads(clean)
        ings = cand.get("ingredients", cand.get("ings", []))
        effs_list = cand.get("effectiveness", cand.get("eff", []))
        
        # Process anonymous effectiveness slots
        # We sort them [Positives High -> Low, Zero, Negatives High -> Low] 
        # to ensure comparison is consistent across recipes.
        effs = [float(str(e).replace('%', '')) for i, e in zip(ings, effs_list) if i == -1]
        effs.sort(reverse=True) 
        
        for i, val in enumerate(effs):
            matrix[row_idx, i] = val
            
        if "stats" in cand:
            for s_name, d_val in cand["stats"].items():
                col_idx = num_effs + stat_to_idx[s_name]
                matrix[row_idx, col_idx] = -d_val["min"] if s_name in REQ_STATS else d_val["max"]

    print("Pass 3/3: Running Numba dominance cull...")
    
    # Sorting optimization:
    # To encounter dominant recipes early, we sort by a score where higher = better.
    # Score = (Sum of Absolute Efficiencies) + (Sum of Stats)
    # We use Absolute Efficiency because -100 is "better" than -50.
    eff_scores = np.sum(np.abs(matrix[:, :num_effs]), axis=1)
    stat_scores = np.sum(matrix[:, num_effs:], axis=1)
    sort_order = np.argsort(-(eff_scores + stat_scores))
    
    matrix = matrix[sort_order]
    raw_lines = [raw_lines[i] for i in sort_order]
    
    is_kept = pareto_filter(matrix, num_effs)
    kept_indices = np.where(is_kept)[0]
    
    print(f"Writing {len(kept_indices)} recipes to output...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, idx in enumerate(kept_indices):
            f.write(raw_lines[idx])
            f.write(",\n" if i < len(kept_indices) - 1 else "\n")
        f.write("]")
        
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    cull_recipes("JEWELING_1.json")