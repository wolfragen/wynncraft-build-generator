import json
import numpy as np
from numba import njit, prange, set_num_threads
import time
import os
import itertools

from data.recipe_loader import load_recipes

from data.stats import (
    ALL_STATS,
    STAT_COUNT,
    STAT_INDEX,
    IDX_DURABILITY
)

# =======================
# HARDWARE CONSTANTS
# =======================

NUM_THREADS = 14
CHUNK_SIZE = 500_000

set_num_threads(NUM_THREADS)

SLOT_COUNT = 6


# ============================================================
# DATA PREP
# ============================================================

def load_and_prepare(ingred_path, profession, include_dura=True):

    with open(ingred_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    filtered = []

    for ing in raw:
        if profession not in ing.get("skills", []):
            continue

        pos = ing.get("posMods", {})
        has_pos_mods = any(v != 0 for v in pos.values())
        has_ingred_eff = "ingredEff" in ing.get("ids", {})
        item_ids = ing.get("itemIDs", {})

        dura = item_ids.get("dura", 0)

        if not (has_pos_mods or has_ingred_eff or (include_dura and dura > 0)):
            continue

        filtered.append(ing)

    num_ingreds = len(filtered)

    eff_bonus = np.zeros(num_ingreds, dtype=np.int16)
    pos_mods = np.zeros((num_ingreds, 6), dtype=np.int16)
    dura_vals = np.zeros(num_ingreds, dtype=np.int16)
    ing_stats = np.zeros((num_ingreds, STAT_COUNT), dtype=np.int16)

    for i, ing in enumerate(filtered):

        ids = ing.get("ids", {})
        pos = ing.get("posMods", {})
        item_ids = ing.get("itemIDs", {})

        # ---- effectiveness bonus
        eff_val = ids.get("ingredEff", 0)
        if isinstance(eff_val, dict):
            eff_bonus[i] = eff_val.get("minimum", 0)
        else:
            eff_bonus[i] = eff_val

        # ---- positional mods
        pos_mods[i,0] = pos.get("right",0)
        pos_mods[i,1] = pos.get("left",0)
        pos_mods[i,2] = pos.get("above",0)
        pos_mods[i,3] = pos.get("under",0)
        pos_mods[i,4] = pos.get("touching",0)
        pos_mods[i,5] = pos.get("notTouching",0)

        # ---- durability
        dura = item_ids.get("dura", 0)
        dura_vals[i] = dura
        if dura != 0:
            ing_stats[i, IDX_DURABILITY] += dura

        # ---- id stats
        for stat_name, value in ids.items():
            if stat_name == "ingredEff":
                continue
            if stat_name in STAT_INDEX:
                idx = STAT_INDEX[stat_name]
                if isinstance(value, dict):
                    ing_stats[i, idx] += value.get("minimum", 0)
                else:
                    ing_stats[i, idx] += value

        # ---- requirement stats
        for stat_name, value in item_ids.items():
            if stat_name in STAT_INDEX:
                idx = STAT_INDEX[stat_name]
                ing_stats[i, idx] += value

    return eff_bonus, pos_mods, dura_vals, ing_stats


# ============================================================
# ADDITIVE INFLUENCE
# ============================================================

def build_additive_influence(pos_mods):

    adj = {
        0:[1,2],1:[0,3],2:[0,3,4],
        3:[1,2,5],4:[2,5],5:[3,4]
    }

    influence = np.zeros((6,6,6), dtype=np.int8)

    for src in range(6):
        for dst in range(6):
            if src == dst:
                continue

            if src%2==0 and dst==src+1:
                influence[src,dst,0] = 1
            if src%2==1 and dst==src-1:
                influence[src,dst,1] = 1

            column = [0,2,4] if src%2==0 else [1,3,5]
            if dst in column:
                if dst > src:
                    influence[src,dst,3] = 1
                elif dst < src:
                    influence[src,dst,2] = 1

            if dst in adj[src]:
                influence[src,dst,4] = 1
            else:
                influence[src,dst,5] = 1

    n_ing = pos_mods.shape[0]
    additive = np.zeros((6,6,n_ing), dtype=np.int16)

    for src in range(6):
        for dst in range(6):
            mask = influence[src,dst]
            for ing in range(n_ing):
                total = 0
                for m in range(6):
                    if mask[m] == 1:
                        total += pos_mods[ing,m]
                additive[src,dst,ing] = total

    return additive


# ============================================================
# PARALLEL ENUMERATOR
# ============================================================

@njit(parallel=True)
def fill_chunk_parallel(
    positions,
    n,
    eff_bonus,
    additive,
    ing_stats,
    dura_vals,
    start_index,
    count,
    out_grid,
    out_eff,
    out_stats,
    out_dura
):

    k = len(positions)

    for idx in prange(count):

        tmp = start_index + idx

        grid = out_grid[idx]
        eff = out_eff[idx]
        stats = out_stats[idx]

        # reset
        for i in range(6):
            grid[i] = -1
            eff[i] = 0

        for s in range(STAT_COUNT):
            stats[s] = 0

        total_dura = 0

        # decode base-n digits
        for i in range(k):
            digit = tmp % n
            tmp //= n
            pos = positions[i]
            grid[pos] = digit
            total_dura += dura_vals[digit]

        # initialize efficiency
        for i in range(6):
            if grid[i] != -1:
                eff[i] = 100 + eff_bonus[grid[i]]

        # accumulate stats
        for i in range(6):
            ing = grid[i]
            if ing == -1:
                continue

            for s in range(STAT_COUNT):
                stats[s] += ing_stats[ing, s]

        # additive influence
        for src in range(6):
            ing = grid[src]
            if ing == -1:
                continue
            for dst in range(6):
                eff[dst] += additive[src,dst,ing]

        for i in range(6):
            if eff[i] < 0:
                eff[i] = 0

        out_dura[idx] = total_dura
        
        
# ============================================================
# MIN DURA FINDER
# ============================================================

def compute_min_dura_for_profession(recipes, profession, lvl_min, lvl_max):
    """
    Compute minimum allowed durability threshold for a profession.

    We find the maximum base durability among all recipes of this
    profession in the given level range.

    Any set consuming more than that is impossible.

    Returns:
        min_dura (negative int)
    """

    max_dura = 0

    for r in recipes:
        if (
            r["skill"] == profession
            and r["lvl"]["minimum"] == lvl_min
            and r["lvl"]["maximum"] == lvl_max
        ):
            dura = r["durability"]["maximum"]
            if dura > max_dura:
                max_dura = dura

    if max_dura == 0:
        raise ValueError("No valid recipes found for profession/level range")

    return -max_dura


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    TARGET_PROFESSION = "JEWELING"
    LEVEL_MIN = 103
    LEVEL_MAX = 105

    DATA_PATH = "data/ingreds_compress.json"
    RECIPES_PATH = "data/recipes_compress.json"

    OUTPUT_DIR = "data/precalc/full"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load recipes
    recipes_data = load_recipes(RECIPES_PATH)

    # Compute durability threshold for profession
    min_dura = compute_min_dura_for_profession(
        recipes_data,
        TARGET_PROFESSION,
        LEVEL_MIN,
        LEVEL_MAX
    )

    print(f"Computed min_dura for {TARGET_PROFESSION}: {min_dura}")

    eff_bonus, pos_mods, dura_vals, ing_stats = load_and_prepare(DATA_PATH, TARGET_PROFESSION, True)

    additive = build_additive_influence(pos_mods)
    n = eff_bonus.shape[0]

    print(f"Total filtered ingredients: {n}")

    for k in range(1, 7):

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{TARGET_PROFESSION}_{k}.json"
        )

        slot_combos = list(itertools.combinations(range(6), k))
        total_recipes = len(slot_combos) * (n ** k)

        print(f"\nGenerating {k}-ingredient recipes")
        print(f"Total recipes: {total_recipes:,}")

        start = time.time()
        total_generated = 0
        total_kept = 0
        checkpoint = max(1, total_recipes // 100)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True

            for positions in slot_combos:

                positions_arr = np.array(positions, dtype=np.int16)
                total_for_combo = n ** k
                start_index = 0

                out_grid = np.zeros((CHUNK_SIZE, 6), dtype=np.int16)
                out_eff = np.zeros((CHUNK_SIZE, 6), dtype=np.int16)
                out_stats = np.zeros((CHUNK_SIZE, STAT_COUNT), dtype=np.int32)
                out_dura = np.zeros(CHUNK_SIZE, dtype=np.int32)

                while start_index < total_for_combo:

                    remaining = total_for_combo - start_index
                    count = min(CHUNK_SIZE, remaining)

                    fill_chunk_parallel(
                        positions_arr,
                        n,
                        eff_bonus,
                        additive,
                        ing_stats,
                        dura_vals,
                        start_index,
                        count,
                        out_grid,
                        out_eff,
                        out_stats,
                        out_dura
                    )

                    lines = []

                    for i in range(count):
                        total_generated += 1

                        if out_dura[i] < min_dura:
                            continue

                        stats_dict = {}

                        for s in range(STAT_COUNT):
                            val = int(out_stats[i, s])
                            if val != 0:
                                stats_dict[ALL_STATS[s]] = val

                        obj = {
                            "ings": [int(x) for x in out_grid[i]],
                            "eff": [int(x) for x in out_eff[i]],
                            "stats": stats_dict
                        }

                        if not first:
                            lines.append(",\n")
                        lines.append(json.dumps(obj))
                        first = False
                        
                        total_kept += 1
                        if total_generated % checkpoint == 0:
                            pct = total_generated / total_recipes * 100
                            elapsed = time.time() - start
                            print(f"{pct:.1f}% | {elapsed:.1f}s", end="\r")

                    f.write("".join(lines))
                    start_index += count

            f.write("\n]")
            
            print(f"Kept after durability prune: {total_kept:,}")
            if total_generated > 0:
                ratio = total_kept / total_generated * 100
                print(f"Kept ratio: {ratio:.2f}%")

        print(f"\nSaved → {output_path}")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        