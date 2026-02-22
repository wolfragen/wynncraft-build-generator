import json
import itertools
import time
import math
import os

def load_raw_data(ingred_path):
    with open(ingred_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_filtered_ingredients(all_ingreds, profession, include_dura):
    """Filters ingredients for Meta properties, self-effectiveness, or positive durability."""
    filtered = []
    for ing in all_ingreds:
        # Check if ingredient is applicable to this skill
        if profession not in ing.get("skills", []):
            continue
        
        pos = ing.get("posMods", {})
        
        # Rule: Meta ingredients have position modifiers OR self-modifying ingredEff
        has_pos_mods = any(v != 0 for v in pos.values())
        has_ingred_eff = "ingredEff" in ing.get("ids", {})
        is_meta = has_pos_mods or has_ingred_eff
        
        # Rule: Positive durability ingredients
        dura = ing.get("itemIDs", {}).get("dura", 0)
        is_dura = dura > 0
        
        if is_meta or (include_dura and is_dura):
            # Pre-extract max stats to speed up the main loop
            stats = {}
            for stat_name, val in ing.get("ids", {}).items():
                if isinstance(val, dict):
                    stats[stat_name] = val.get("maximum", val.get("max", 0))
                else:
                    stats[stat_name] = val
            
            # Add durability to the stats dict for easier processing
            if dura != 0:
                stats["dura"] = dura
            
            filtered.append({
                "id": ing["id"],
                "name": ing.get("displayName", ing.get("name", "Unknown")),
                "posMods": pos,
                "stats": stats
            })
    return filtered

def calculate_recipe_stats(grid):
    """Calculates final effectiveness and summed stats for a specific 6-slot grid."""
    
    # Base effectiveness for all 6 slots is 100%
    effs = [100] * 6
    
    # 4-way adjacency map (Up, Down, Left, Right)
    adjacents = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3, 4],
        3: [1, 2, 5],
        4: [2, 5],
        5: [3, 4]
    }

    # 1. Sum up all effectiveness modifiers per slot
    for i, ing in enumerate(grid):
        if ing is None: continue
        
        pm = ing["posMods"]
        
        # Self-modifying stat (e.g., Doom Stone)
        effs[i] += ing["stats"].get("ingredEff", 0)
        
        # Left / Right (Propagates to the opposite column in the same row)
        if i % 2 == 0: # Left Column
            effs[i + 1] += pm.get('right', 0)
        else:          # Right Column
            effs[i - 1] += pm.get('left', 0)
            
        # Above / Under (Propagates to the entire column)
        column = [0, 2, 4] if i % 2 == 0 else [1, 3, 5]
        for target in column:
            if target > i:
                effs[target] += pm.get('under', 0)
            elif target < i:
                effs[target] += pm.get('above', 0)
                
        # Touching (4-way neighbors only)
        for target in adjacents[i]:
            effs[target] += pm.get('touching', 0)
            
        # Not Touching (Every slot except self and 4-way neighbors)
        for target in range(6):
            if target != i and target not in adjacents[i]:
                effs[target] += pm.get('notTouching', 0)

    # Floor all effectiveness at 0%
    eff_values = [max(0, e) for e in effs]
    
    # 2. Apply effectiveness to ingredient stats and accumulate
    total_stats = {}
    for s, ing in enumerate(grid):
        if ing is None: continue
        multiplier = eff_values[s]
        
        for name, raw_val in ing["stats"].items():
            if name == "ingredEff": 
                continue # Do not output the meta stat itself
                
            if name == "dura":
                # Durability is untouched by effectiveness modifiers
                total_stats[name] = total_stats.get(name, 0) + raw_val
            else:
                # Apply rounding: (raw * effectiveness) // 100
                boosted_val = (raw_val * multiplier) // 100
                total_stats[name] = total_stats.get(name, 0) + boosted_val
            
    return {
        "ingredients": [ing["id"] if ing else -1 for ing in grid],
        "effectiveness": [f"{e}%" for e in eff_values],
        "stats": total_stats
    }

def run_precalculation(profession, include_dura_ingredients=True):
    """Main execution loop for the project."""
    print(f"--- PRECALCULATING: {profession} ---")
    
    # Define paths
    data_dir = "data"
    precalc_dir = os.path.join(data_dir, "precalc")
    ingred_path = os.path.join(data_dir, "ingreds_compress.json")
    
    os.makedirs(precalc_dir, exist_ok=True)
    
    # Load and filter
    all_data = load_raw_data(ingred_path)
    filtered = get_filtered_ingredients(all_data, profession, include_dura_ingredients)
    print(f"Relevant ingredients found: {len(filtered)}")
    
    # Generate files PROFESSION_META_1.json to PROFESSION_META_6.json
    for i in range(1, 7):
        filename = os.path.join(precalc_dir, f"{profession}_META_{i}.json")
        
        # Math: total = (6 choose i) * (N ^ i)
        num_slot_combos = math.comb(6, i)
        total_for_file = num_slot_combos * (len(filtered) ** i)
        
        print(f"\nStarting {profession}_META_{i}.json (Total: {total_for_file:,} recipes)")
        start_time = time.time()
        processed = 0
        checkpoint = max(1, total_for_file // 100) # Print progress every 1%
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("[\n")
            first_entry = True
            
            # Step A: Choose which slots in the 6-slot grid will be filled
            for active_slot_indices in itertools.combinations(range(6), i):
                
                # Step B: Fill those slots with every possible combination of our filtered ingredients
                for ing_combo in itertools.product(filtered, repeat=i):
                    # Construct the 6-slot grid (None = -1/Empty)
                    grid = [None] * 6
                    for slot_local_idx, grid_idx in enumerate(active_slot_indices):
                        grid[grid_idx] = ing_combo[slot_local_idx]
                    
                    # Compute stats
                    recipe_data = calculate_recipe_stats(grid)
                    
                    # Stream to file
                    if not first_entry: f.write(",\n")
                    f.write(json.dumps(recipe_data))
                    first_entry = False
                    
                    # Progress update
                    processed += 1
                    if processed % checkpoint == 0:
                        pct = (processed * 100) // total_for_file
                        elapsed = time.time() - start_time
                        print(f"Progress: {pct}% | {processed}/{total_for_file} | {elapsed:.0f}s elapsed", end='\r')
            
            f.write("\n]")
        
        total_time = time.time() - start_time
        print(f"\nFinished {profession}_META_{i}.json in {total_time:.2f} seconds.")

def main():
    # You can change the profession here
    TARGET_PROFESSION = "JEWELING" 
    
    if not os.path.exists("data/ingreds_compress.json"):
        print("Error: 'data/ingreds_compress.json' not found.")
    else:
        run_precalculation(TARGET_PROFESSION, include_dura_ingredients=True)

