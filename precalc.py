import json
import itertools
import time
import math
import os

def load_raw_data(ingred_path):
    with open(ingred_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_filtered_ingredients(all_ingreds, profession, include_dura):
    """Filters ingredients and pre-extracts min/max for all relevant stats."""
    filtered = []
    for ing in all_ingreds:
        if profession not in ing.get("skills", []):
            continue
        
        pos = ing.get("posMods", {})
        has_pos_mods = any(v != 0 for v in pos.values())
        has_ingred_eff = "ingredEff" in ing.get("ids", {})
        is_meta = has_pos_mods or has_ingred_eff
        
        # Collect Item IDs (Requirements and Durability)
        item_ids = ing.get("itemIDs", {})
        # Collect Consumable IDs (Duration, Charges, etc)
        cons_ids = ing.get("consumableIDs", {})
        
        dura = item_ids.get("dura", 0)
        is_dura = dura > 0
        
        if is_meta or (include_dura and is_dura):
            stats = {}
            # 1. Process regular IDs (affected by effectiveness)
            for name, val in ing.get("ids", {}).items():
                if isinstance(val, dict):
                    stats[name] = {"min": val.get("minimum", 0), "max": val.get("maximum", 0)}
                else:
                    stats[name] = {"min": val, "max": val}
            
            # 2. Process Requirements (affected by effectiveness)
            reqs = ["strReq", "dexReq", "intReq", "defReq", "agiReq"]
            for r in reqs:
                val = item_ids.get(r, 0)
                if val != 0:
                    stats[r] = {"min": val, "max": val}

            # 3. Process Non-affected stats (Static)
            static_stats = {}
            if dura != 0: static_stats["dura"] = {"min": dura, "max": dura}
            
            for s in ["duration", "charges"]:
                val = cons_ids.get(s, 0)
                if val != 0:
                    static_stats[s] = {"min": val, "max": val}
            
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
        # ingredEff is stored in 'stats' as min/max (usually identical)
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

    eff_values = [max(0, e) for e in effs]
    
    total_stats = {}
    
    for s, ing in enumerate(grid):
        if ing is None: continue
        multiplier = eff_values[s]
        
        # A. Process Affected Stats
        for name, range_val in ing["stats"].items():
            if name == "ingredEff": continue
            
            # (raw * effectiveness) // 100
            b_min = (range_val["min"] * multiplier) // 100
            b_max = (range_val["max"] * multiplier) // 100
            
            if b_min != 0 or b_max != 0:
                if name not in total_stats: total_stats[name] = {"min": 0, "max": 0}
                total_stats[name]["min"] += b_min
                total_stats[name]["max"] += b_max
        
        # B. Process Static Stats (Durability, etc)
        for name, range_val in ing["static_stats"].items():
            if name not in total_stats: total_stats[name] = {"min": 0, "max": 0}
            total_stats[name]["min"] += range_val["min"]
            total_stats[name]["max"] += range_val["max"]
            
    return {
        "ings": [ing["id"] if ing else -1 for ing in grid],
        "eff": eff_values,
        "stats": total_stats
    }

def run_precalculation(profession, include_dura_ingredients=True):
    print(f"--- PRECALCULATING: {profession} ---")
    data_dir = "data"
    precalc_dir = os.path.join(data_dir, "precalc/full")
    ingred_path = os.path.join(data_dir, "ingreds_compress.json")
    os.makedirs(precalc_dir, exist_ok=True)
    
    all_data = load_raw_data(ingred_path)
    filtered = get_filtered_ingredients(all_data, profession, include_dura_ingredients)
    print(f"Relevant ingredients found: {len(filtered)}")
    
    for i in range(1, 7):
        filename = os.path.join(precalc_dir, f"{profession}_META_{i}.json")
        num_slot_combos = math.comb(6, i)
        total_for_file = num_slot_combos * (len(filtered) ** i)
        
        print(f"\nStarting {profession}_META_{i}.json (Total: {total_for_file:,} recipes)")
        start_time = time.time()
        processed = 0
        checkpoint = max(1, total_for_file // 100)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[\n")
            first_entry = True
            for active_slot_indices in itertools.combinations(range(6), i):
                for ing_combo in itertools.product(filtered, repeat=i):
                    grid = [None] * 6
                    for slot_local_idx, grid_idx in enumerate(active_slot_indices):
                        grid[grid_idx] = ing_combo[slot_local_idx]
                    
                    recipe_data = calculate_recipe_stats(grid)
                    
                    if not first_entry: f.write(",\n")
                    f.write(json.dumps(recipe_data))
                    first_entry = False
                    
                    processed += 1
                    if processed % checkpoint == 0:
                        pct = (processed * 100) // total_for_file
                        print(f"Progress: {pct}% | {processed}/{total_for_file} | {time.time()-start_time:.0f}s", end='\r')
            f.write("\n]")
        print(f"\nFinished {profession}_META_{i}.json in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    TARGET_PROFESSION = "JEWELING" 
    if os.path.exists("data/ingreds_compress.json"):
        run_precalculation(TARGET_PROFESSION, include_dura_ingredients=True)