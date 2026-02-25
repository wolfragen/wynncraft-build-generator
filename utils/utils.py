import numpy as np
from data.stats import STAT_INDEX, STAT_COUNT
from core.leaf_evaluator import evaluate_leaf


def craft(
    ingredient_names,
    recipe,
    ingredients_raw,
):
    """
    Compute ALL stats of a craft using raw precomputed stat arrays.
    """

    if len(ingredient_names) != 6:
        raise ValueError("Craft must contain exactly 6 ingredients.")

    # ------------------------------------------------------------
    # Lookup raw ingredients
    # ------------------------------------------------------------
    name_to_raw = {
        ing["name"]: ing
        for ing in ingredients_raw
    }

    raw_ings = [name_to_raw[name] for name in ingredient_names]

    # ------------------------------------------------------------
    # Compute effectiveness (correct Wynn logic)
    # ------------------------------------------------------------
    eff = np.ones(6, dtype=np.int32) * 100
    
    def slot_to_rc(slot):
        return slot // 2, slot % 2
    
    def rc_to_slot(r, c):
        return r * 2 + c
    
    for slot in range(6):
    
        r, c = slot_to_rc(slot)
        pos_mods = raw_ings[slot].get("posMods", {})
    
        # Precompute touching neighbors
        touching = []
    
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 2:
                touching.append(rc_to_slot(nr, nc))
    
        all_slots = set(range(6))
    
        for mod, value in pos_mods.items():
    
            # ABOVE → same column, strictly above
            if mod == "above":
                for rr in range(r):
                    target = rc_to_slot(rr, c)
                    eff[target] += value
    
            # UNDER → same column, strictly below
            elif mod == "under":
                for rr in range(r + 1, 3):
                    target = rc_to_slot(rr, c)
                    eff[target] += value
    
            # LEFT → entire column to the left
            elif mod == "left" and c > 0:
                for rr in range(3):
                    target = rc_to_slot(rr, c - 1)
                    eff[target] += value
    
            # RIGHT → entire column to the right
            elif mod == "right" and c < 1:
                for rr in range(3):
                    target = rc_to_slot(rr, c + 1)
                    eff[target] += value
    
            # TOUCHING → orthogonal neighbors only
            elif mod == "touching":
                for target in touching:
                    eff[target] += value
    
            # NOT TOUCHING → all slots except touching ones
            elif mod == "notTouching":
                non_touching = all_slots - set(touching) - set([slot])
                for target in non_touching:
                    eff[target] += value

    # ------------------------------------------------------------
    # Build stat matrices directly from raw arrays
    # ------------------------------------------------------------
    stat_min_matrix = np.stack([ing["stats_min"] for ing in raw_ings]).astype(np.int32)
    stat_max_matrix = np.stack([ing["stats_max"] for ing in raw_ings]).astype(np.int32)

    # ------------------------------------------------------------
    # Recipe base
    # ------------------------------------------------------------
    base_min = np.zeros(STAT_COUNT, dtype=np.int32)
    base_max = np.zeros(STAT_COUNT, dtype=np.int32)

    dur_idx = STAT_INDEX.get("durability")
    if dur_idx is not None:
        base_min[dur_idx] += recipe.scaled_dura_min
        base_max[dur_idx] += recipe.scaled_dura_max

    # ------------------------------------------------------------
    # Dummy masks (no constraints)
    # ------------------------------------------------------------
    has_min_mask = np.zeros(STAT_COUNT, dtype=np.bool_)
    has_max_mask = np.zeros(STAT_COUNT, dtype=np.bool_)
    min_vals = np.zeros(STAT_COUNT, dtype=np.int32)
    max_vals = np.zeros(STAT_COUNT, dtype=np.int32)
    weights = np.zeros(STAT_COUNT, dtype=np.float32)

    ingredient_indices = np.arange(6, dtype=np.int32)

    score, acc_min, acc_max = evaluate_leaf(
        ingredient_indices,
        6,
        stat_min_matrix,
        stat_max_matrix,
        base_min,
        base_max,
        eff,
        True,
        dur_idx,
        has_min_mask,
        has_max_mask,
        min_vals,
        max_vals,
        weights,
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------
    print("ing:", ingredient_names)
    print("eff:", eff.tolist())
    print()

    for stat_name, idx in STAT_INDEX.items():
        if acc_min[idx] != 0 or acc_max[idx] != 0:
            print(f"{stat_name}: {acc_min[idx]}/{acc_max[idx]}")