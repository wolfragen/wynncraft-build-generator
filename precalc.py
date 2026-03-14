import json
import itertools
import time
import math
import os

from data.recipe_loader import load_recipes
from data.stats import CONSU_SKILLS

def load_raw_data(ingred_path):
    with open(ingred_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_filtered_ingredients(all_ingreds, profession, include_dura, dura_str):
    """Filters ingredients and pre-extracts min/max for all relevant stats."""
    filtered = []
    for ing in all_ingreds:
        if profession not in ing.get("skills", []):
            continue
        
        pos = ing.get("posMods", {})
        has_pos_mods = any(v != 0 for v in pos.values())
        has_ingred_eff = "ingredEff" in ing.get("ids", {})
        
        item_ids = ing.get("itemIDs", {})
        cons_ids = ing.get("consumableIDs", {})
        charges = cons_ids.get("charges", 0)
        
        is_meta = has_pos_mods or has_ingred_eff or charges != 0
        
        durability = item_ids.get("dura", 0)
        duration = cons_ids.get("dura", 0)
        dura = durability
        if dura_str == "duration":
            dura = duration
        is_dura = dura > 0
        
        if is_meta or (include_dura and is_dura):
            stats = {}
            for name, val in ing.get("ids", {}).items():
                if isinstance(val, dict):
                    stats[name] = {"min": val.get("minimum", 0), "max": val.get("maximum", 0)}
                else:
                    stats[name] = {"min": val, "max": val}
            
            reqs = ["strReq", "dexReq", "intReq", "defReq", "agiReq"]
            for r in reqs:
                val = item_ids.get(r, 0)
                if val != 0:
                    stats[r] = {"min": val, "max": val}

            static_stats = {}
            if dura != 0: static_stats[dura_str] = {"min": dura, "max": dura}
            
            if charges != 0:
                static_stats["charges"] = {"min": charges, "max": charges}
            
            filtered.append({
                "id": ing["id"],
                "name": ing.get("displayName", ing.get("name", "Unknown")),
                "posMods": pos,
                "stats": stats,
                "static_stats": static_stats
            })
    return filtered

def calculate_recipe_stats(grid):
    """Calculates final effectiveness and summed min/max stats."""
    effs = [100] * 6
    adjacents = {0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 5], 4: [2, 5], 5: [3, 4]}

    for i, ing in enumerate(grid):
        if ing is None: continue
        pm = ing["posMods"]
        effs[i] += ing["stats"].get("ingredEff", {"min": 0})["min"]
        
        if i % 2 == 0: effs[i + 1] += pm.get('right', 0)
        else: effs[i - 1] += pm.get('left', 0)
            
        column = [0, 2, 4] if i % 2 == 0 else [1, 3, 5]
        for target in column:
            if target > i: effs[target] += pm.get('under', 0)
            elif target < i: effs[target] += pm.get('above', 0)
                
        for target in adjacents[i]: effs[target] += pm.get('touching', 0)
        for target in range(6):
            if target != i and target not in adjacents[i]:
                effs[target] += pm.get('notTouching', 0)

    # Effectiveness is NO LONGER clamped to 0 so negative effs evaluate properly.
    total_stats = {}
    
    for s, ing in enumerate(grid):
        if ing is None: continue
        multiplier = effs[s]
        
        for name, range_val in ing["stats"].items():
            if name == "ingredEff": continue
            b_min = (range_val["min"] * multiplier) // 100
            b_max = (range_val["max"] * multiplier) // 100
            
            if b_min != 0 or b_max != 0:
                if name not in total_stats: total_stats[name] = {"min": 0, "max": 0}
                total_stats[name]["min"] += b_min
                total_stats[name]["max"] += b_max
        
        for name, range_val in ing["static_stats"].items():
            if name not in total_stats: total_stats[name] = {"min": 0, "max": 0}
            total_stats[name]["min"] += range_val["min"]
            total_stats[name]["max"] += range_val["max"]
            
    return {
        "ings": [ing["id"] if ing else -1 for ing in grid],
        "eff": effs,
        "stats": total_stats
    }

def compare_recipes(r_a, r_b):
    """Returns 1 if a > b, -1 if b > a, 2 if a == b, 0 if incomparable"""
    a_better = False
    b_better = False
    
    # 1. Compare Effectiveness on Empty Slots (-1 ingredients)
    empty_effs_a = [r_a['eff'][i] for i in range(6) if r_a['ings'][i] == -1]
    empty_effs_b = [r_b['eff'][i] for i in range(6) if r_b['ings'][i] == -1]
    
    pos_a = sorted([e for e in empty_effs_a if e > 0], reverse=True)
    pos_b = sorted([e for e in empty_effs_b if e > 0], reverse=True)
    
    neg_a = sorted([e for e in empty_effs_a if e < 0]) # Ascending: -50, -10
    neg_b = sorted([e for e in empty_effs_b if e < 0])
    
    # If the counts of positive or negative effs differ, they are incomparable
    if len(pos_a) != len(pos_b) or len(neg_a) != len(neg_b):
        return 0
        
    for pa, pb in zip(pos_a, pos_b):
        if pa > pb: a_better = True
        elif pb > pa: b_better = True
        
    for na, nb in zip(neg_a, neg_b):
        # Lower negative is better (-50 is better than -10)
        if na < nb: a_better = True
        elif nb < na: b_better = True

    if a_better and b_better:
        return 0

    # 2. Compare Stats
    stats_a = r_a['stats']
    stats_b = r_b['stats']
    req_stats = {"strReq", "dexReq", "intReq", "defReq", "agiReq"}
    keys = set(stats_a.keys()).union(set(stats_b.keys()))
    
    for k in keys:
        a_val = stats_a.get(k, {"min": 0, "max": 0})
        b_val = stats_b.get(k, {"min": 0, "max": 0})
        
        a_min, a_max = a_val["min"], a_val["max"]
        b_min, b_max = b_val["min"], b_val["max"]
        
        if k in req_stats:
            # For requirements, lower is better
            if a_min < b_min: a_better = True
            elif b_min < a_min: b_better = True
            
            if a_max < b_max: a_better = True
            elif b_max < a_max: b_better = True
        else:
            # For standard stats, higher is better
            if a_min > b_min: a_better = True
            elif b_min > a_min: b_better = True
            
            if a_max > b_max: a_better = True
            elif b_max > a_max: b_better = True
            
        # Early exit if incomparable
        if a_better and b_better:
            return 0 
            
    if a_better: return 1
    if b_better: return -1
    return 2 # Identical

def local_cull(recipes):
    """Keeps only the Pareto frontier of recipes within a given localized batch."""
    survivors = []
    for r in recipes:
        is_dominated = False
        to_remove = []
        for i, s in enumerate(survivors):
            cmp = compare_recipes(r, s)
            if cmp == -1 or cmp == 2:
                # 's' strictly dominates 'r', or they are completely identical. 
                is_dominated = True
                break
            elif cmp == 1:
                # 'r' strictly dominates 's'
                to_remove.append(i)
        
        if not is_dominated:
            # Remove dominated survivors in reverse order to preserve indices
            for i in reversed(to_remove):
                survivors.pop(i)
            survivors.append(r)
            
    return survivors

def get_unique_permutations(ing_multiset):
    """Yields only unique permutations of a multiset based on ingredient IDs."""
    seen = set()
    for perm in itertools.permutations(ing_multiset):
        perm_ids = tuple(ing['id'] for ing in perm)
        if perm_ids not in seen:
            seen.add(perm_ids)
            yield perm

def run_precalculation(profession, include_dura_ingredients=True, pre_filled=5, min_dura=None):
    
    dura = "durability"
    if profession in CONSU_SKILLS:
        dura = "duration"
    
    should_check_dura = min_dura is not None
    print(f"--- PRECALCULATING: {profession} ---")
    data_dir = "data"
    precalc_dir = os.path.join(data_dir, "precalc/full")
    ingred_path = os.path.join(data_dir, "ingreds_compress.json")
    os.makedirs(precalc_dir, exist_ok=True)
    
    all_data = load_raw_data(ingred_path)
    filtered = get_filtered_ingredients(all_data, profession, include_dura_ingredients, dura_str=dura)
    print(f"Relevant ingredients found: {len(filtered)}")
    
    for i in range(1, pre_filled+1):
        filename = os.path.join(precalc_dir, f"{profession}_META_{i}.json")
        
        num_multisets = math.comb(len(filtered) + i - 1, i)
        num_slot_combos = math.comb(6, i)
        total_generated_expected = num_slot_combos * (len(filtered) ** i)
        
        print(f"\nStarting {profession}_META_{i}.json")
        print(f"Expected generated combinations: {total_generated_expected:,}")
        
        start_time = time.time()
        total_generated = 0
        total_saved = 0
        total_culled = 0
        
        checkpoint = max(1, num_multisets // 100)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[\n")
            first_entry = True
            
            for multiset_idx, ing_multiset in enumerate(itertools.combinations_with_replacement(filtered, i)):
                local_recipes = []
                
                for ing_perm in get_unique_permutations(ing_multiset):
                    for active_slot_indices in itertools.combinations(range(6), i):
                        grid = [None] * 6
                        for slot_local_idx, grid_idx in enumerate(active_slot_indices):
                            grid[grid_idx] = ing_perm[slot_local_idx]
                        
                        recipe_data = calculate_recipe_stats(grid)
                        total_generated += 1
                        
                        # --- dura prune ---
                        if should_check_dura:
                            dura_stat = recipe_data.get("stats", {}).get(dura)
                            if dura_stat is not None:
                                if dura_stat["max"] < min_dura:
                                    total_culled += 1
                                    continue
                        
                        local_recipes.append(recipe_data)
                
                survivors = local_cull(local_recipes)
                
                total_culled += (len(local_recipes) - len(survivors))
                total_saved += len(survivors)
                
                for recipe in survivors:
                    if not first_entry: f.write(",\n")
                    f.write(json.dumps(recipe))
                    first_entry = False
                
                if (multiset_idx + 1) % checkpoint == 0 or (multiset_idx + 1) == num_multisets:
                    pct = ((multiset_idx + 1) * 100) // num_multisets
                    elapsed = time.time() - start_time
                    print(f"Progress: {pct}% | "
                          f"Gen: {total_generated:,} | Saved: {total_saved:,} | Culled: {total_culled:,} "
                          f"| Time: {elapsed:.0f}s", end='\r')
            
            f.write("\n]")
            
        print(f"\nFinished {profession}_META_{i}.json in {time.time() - start_time:.2f}s")
        print(f"Final Batch Stats - Saved: {total_saved:,} | Culled: {total_culled:,} ", end="")
        if total_generated > 0:
            print(f"({(total_culled / total_generated) * 100:.2f}% size reduction)")
            
            
# ============================================================ 
# MIN DURA FINDER 
# ============================================================ 
def compute_min_dura_for_profession(recipes, profession, lvl_min, lvl_max): 
    """ Compute minimum allowed dura threshold for a profession. 
    We find the maximum base dura among all recipes of this profession in the given level range. 
    Any set consuming more than that is impossible. Returns: min_dura (negative int) 
    """ 
    
    dura_str = "durability"
    if profession in CONSU_SKILLS:
        dura_str = "duration"
        
    max_dura = 0 
    for r in recipes: 
        r = r.data
        if ( r["skill"] == profession and r["lvl"]["minimum"] == lvl_min and r["lvl"]["maximum"] == lvl_max ): 
            dura = r[dura_str]["maximum"] 
            if dura > max_dura: 
                max_dura = dura
    if max_dura == 0: 
        raise ValueError("No valid recipes found for profession/level range") 
    return -max_dura


if __name__ == "__main__":
    professions = ["ALCHEMISM", "SCRIBING", "COOKING"]
    if os.path.exists("data/ingreds_compress.json"):
        for TARGET_PROFESSION in professions:
        
            LEVEL_MIN = 103 
            LEVEL_MAX = 105
            RECIPES_PATH = "data/recipes_compress.json" 
            recipes_data = load_recipes(RECIPES_PATH) # Compute dura threshold for profession 
            min_dura = compute_min_dura_for_profession( recipes_data, TARGET_PROFESSION, LEVEL_MIN, LEVEL_MAX )
            run_precalculation(TARGET_PROFESSION, include_dura_ingredients=True, pre_filled=5, min_dura=min_dura)
    else:
        print("Data file not found. Ensure 'data/ingreds_compress.json' exists.")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        