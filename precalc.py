import json
import numpy as np
from numba import njit
import time
import os
import itertools


SLOT_COUNT = 6


# ============================================================
# DATA PREP
# ============================================================

def load_and_prepare(ingred_path, profession, include_dura=True):
    with open(ingred_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    filtered = []
    stat_names = set()

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

        for k in ing.get("ids", {}).keys():
            if k != "ingredEff":
                stat_names.add(k)

        for r in ["strReq","dexReq","intReq","defReq","agiReq"]:
            if item_ids.get(r,0) != 0:
                stat_names.add(r)

        if dura != 0:
            stat_names.add("durability")

    stat_list = sorted(stat_names)
    stat_index = {s:i for i,s in enumerate(stat_list)}

    num_stats = len(stat_list)
    num_ingreds = len(filtered)

    stat_min = np.zeros((num_ingreds, num_stats), dtype=np.int16)
    stat_max = np.zeros((num_ingreds, num_stats), dtype=np.int16)
    eff_bonus = np.zeros(num_ingreds, dtype=np.int16)
    pos_mods = np.zeros((num_ingreds, 6), dtype=np.int16)

    for i, ing in enumerate(filtered):
        ids = ing.get("ids", {})
        item_ids = ing.get("itemIDs", {})
        pos = ing.get("posMods", {})

        eff_val = ids.get("ingredEff", 0)
        if isinstance(eff_val, dict):
            eff_bonus[i] = eff_val.get("minimum", 0)
        else:
            eff_bonus[i] = eff_val

        pos_mods[i,0] = pos.get("right",0)
        pos_mods[i,1] = pos.get("left",0)
        pos_mods[i,2] = pos.get("above",0)
        pos_mods[i,3] = pos.get("under",0)
        pos_mods[i,4] = pos.get("touching",0)
        pos_mods[i,5] = pos.get("notTouching",0)

        for name,val in ids.items():
            if name == "ingredEff":
                continue
            idx = stat_index[name]
            if isinstance(val, dict):
                stat_min[i,idx] = val.get("minimum",0)
                stat_max[i,idx] = val.get("maximum",0)
            else:
                stat_min[i,idx] = val
                stat_max[i,idx] = val

        for r in ["strReq","dexReq","intReq","defReq","agiReq"]:
            v = item_ids.get(r,0)
            if v != 0:
                idx = stat_index[r]
                stat_min[i,idx] = v
                stat_max[i,idx] = v

        dura = item_ids.get("dura",0)
        if dura != 0:
            idx = stat_index["durability"]
            stat_min[i,idx] = dura
            stat_max[i,idx] = dura

    return stat_min, stat_max, eff_bonus, pos_mods, stat_list


# ============================================================
# INFLUENCE MATRIX
# ============================================================

def build_influence_matrix():
    influence = np.zeros((6,6,6), dtype=np.int8)

    adj = {
        0:[1,2],1:[0,3],2:[0,3,4],
        3:[1,2,5],4:[2,5],5:[3,4]
    }

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

    return influence


# ============================================================
# NUMBA CORE FOR ONE RECIPE
# ============================================================

@njit
def compute_recipe(grid, eff, stat_min, stat_max, eff_bonus, pos_mods, influence):
    for i in range(6):
        if grid[i] == -1:
            eff[i] = 0
        else:
            eff[i] = 100 + eff_bonus[grid[i]]

    for src in range(6):
        ing = grid[src]
        if ing == -1:
            continue
        for dst in range(6):
            for m in range(6):
                if influence[src,dst,m] == 1:
                    eff[dst] += pos_mods[ing,m]

    for i in range(6):
        if eff[i] < 0:
            eff[i] = 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    TARGET_PROFESSION = "TAILORING"
    DATA_PATH = "data/ingreds_compress.json"
    OUTPUT_DIR = "data/precalc/full"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stat_min, stat_max, eff_bonus, pos_mods, stat_list = \
        load_and_prepare(DATA_PATH, TARGET_PROFESSION, True)

    influence = build_influence_matrix()
    n = stat_min.shape[0]

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
        processed = 0
        checkpoint = max(1, total_recipes // 100)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True

            for positions in slot_combos:

                for ing_combo in itertools.product(range(n), repeat=k):

                    grid = np.full(6, -1, dtype=np.int16)
                    eff = np.zeros(6, dtype=np.int16)

                    for idx, pos in enumerate(positions):
                        grid[pos] = ing_combo[idx]

                    compute_recipe(grid, eff,
                                   stat_min, stat_max,
                                   eff_bonus, pos_mods,
                                   influence)

                    obj = {
                        "ings": [int(x) for x in grid],
                        "eff": [int(x) for x in eff],
                        "stats": {}
                    }

                    if not first:
                        f.write(",\n")
                    f.write(json.dumps(obj))
                    first = False

                    processed += 1
                    if processed % checkpoint == 0:
                        pct = processed / total_recipes * 100
                        elapsed = time.time() - start
                        print(f"{pct:.1f}% | {elapsed:.1f}s", end="\r")

            f.write("\n]")

        print(f"\nSaved → {output_path}")