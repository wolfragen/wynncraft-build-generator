"""
validate_separable.py — validate the separable solver against the DFS.

For each case: get the DFS optimum (from a cached fixture if present, else run the
DFS once and cache it), run the separable solver, and assert that BOTH builds score
the same through the craft.js oracle (compute_crafted_stats + score_query) and are
both VALID. Comparing oracle scores (not raw ids) is robust to tie/mirror builds.

The DFS fixtures live in tests/fixtures/dfs_results.json so future runs validate
fast without re-running the slow DFS. Delete that file to regenerate.

Run from python/:  python tests/validate_separable.py
"""
import os
import sys
import json
from time import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from data.ingredient_loader import load_ingredients
from data.ingredient_db import build_dual_db
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients, split_and_cull_by_sign
from data.stats import CONSU_SKILLS
from core.search_engine import search_pipelined
from core.warmup import warm_numba
from main_decode import (_index_ingredients, compute_effectiveness,
                         compute_crafted_stats, score_query)
from separable_search import solve_separable
import main_separable

FIXTURES = os.path.join(_HERE, "fixtures", "dfs_results.json")

# (name, skill, item_type, lvl_min, lvl_max, query)
CASES = [
    ("arm_sustain", "ARMOURING", "CHESTPLATE", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "hpBonus": {"weight": 50},
      "durability": {"min": 75, "weight": 1}}),
    ("arm_full", "ARMOURING", "CHESTPLATE", 117, 119, main_separable.build_user_query()),
    ("arm_tightreq", "ARMOURING", "CHESTPLATE", 117, 119,
     dict(main_separable.build_user_query(), strReq={"max": 10, "ingredient_filter": False})),
    ("tai_legs", "TAILORING", "LEGGINGS", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 400}, "spd": {"weight": 400},
      "hpBonus": {"weight": 50}, "durability": {"min": 50, "weight": 1}}),
    ("wea_dagger", "WEAPONSMITHING", "DAGGER", 117, 119,
     {"damPct": {"weight": 1000}, "damRaw": {"weight": 200}, "sdPct": {"weight": 1000},
      "sdRaw": {"weight": 200}, "mr": {"weight": 300}, "durability": {"min": 50, "weight": 1}}),
    ("jew_ring", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"weight": 300},
      "hpBonus": {"weight": 50}, "durability": {"min": 30, "weight": 1}}),
    # tank: binding hpBonus MIN constraint (>=2000) + raw regen
    ("arm_tank", "ARMOURING", "CHESTPLATE", 117, 119,
     {"hpBonus": {"weight": 100, "min": 2000}, "hprRaw": {"weight": 50},
      "mr": {"weight": 500}, "durability": {"min": 75, "weight": 1}}),
    # spell weapon: spell damage + int SP
    ("wea_spell", "WEAPONSMITHING", "DAGGER", 117, 119,
     {"sdPct": {"weight": 1000}, "sdRaw": {"weight": 300}, "int": {"min": 0, "weight": 1500},
      "mr": {"weight": 400}, "intReq": {"max": 45}, "durability": {"min": 50, "weight": 1}}),
    # negative weight (minimise agi) while maximising mr
    ("jew_minimise", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "agi": {"weight": -800}, "durability": {"min": 30, "weight": 1}}),
    # MAX constraint on a non-req stat (spd <= 0)
    ("arm_maxcon", "ARMOURING", "CHESTPLATE", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"max": 0},
      "durability": {"min": 75, "weight": 1}}),
    # tailoring damage + SP + reqs (fuller, harder)
    ("tai_damage", "TAILORING", "LEGGINGS", 117, 119,
     {"damPct": {"weight": 1000}, "damRaw": {"weight": 200}, "sdPct": {"weight": 1000},
      "sdRaw": {"weight": 200}, "str": {"min": 0, "weight": 1500}, "dex": {"min": 0, "weight": 1500},
      "mr": {"weight": 500}, "strReq": {"max": 45}, "dexReq": {"max": 45},
      "durability": {"min": 50, "weight": 1}}),
    # jeweling fuller: int SP + req cap
    ("jew_full", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"weight": 300},
      "hpBonus": {"weight": 50}, "int": {"min": 0, "weight": 800}, "intReq": {"max": 30},
      "durability": {"min": 30, "weight": 1}}),
]


def run_dfs(ings, recipes, skill, item_type, tier, lmn, lmx, query):
    consumable = skill in CONSU_SKILLS
    q = build_query(user_json=query, search_for_inversion=True, item_type=item_type,
                    skill=skill, consumable=consumable, fast_cull=False, full_meta=False)
    rr = find_recipe(recipes=recipes, item_type=item_type, skill=skill, lvl_min=lmn, lvl_max=lmx)
    rec = build_recipe(rr, q, tier=tier)
    filt = filter_raw_ingredients(ings, q, rec, cull=False)
    pos, neg = split_and_cull_by_sign(filt, q, rec)
    ddb = build_dual_db(pos, neg, q)
    best = search_pipelined(skill, q, rec, ddb, max_cull=q.suggested_max_cull)
    return [int(i) for i in best]


def oracle(build, recipe_raw, tier, query, ibid):
    entries = [ibid[i] for i in build]
    eff = compute_effectiveness(entries)
    crafted = compute_crafted_stats(recipe_raw.data, tier, entries, eff)
    r = score_query(crafted, query)
    return r["score"], r["valid"]


def main():
    os.makedirs(os.path.dirname(FIXTURES), exist_ok=True)
    fixtures = {}
    if os.path.exists(FIXTURES):
        with open(FIXTURES, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

    ings = load_ingredients("data/ingreds_compress.json")
    _, ibid = _index_ingredients("data/ingreds_compress.json")
    recipes = load_recipes("data/recipes_compress.json")
    warmed = False

    passed = failed = 0
    summary = []
    for name, skill, itype, lmn, lmx, query in CASES:
        tier = 3
        rr = find_recipe(recipes=recipes, item_type=itype, skill=skill, lvl_min=lmn, lvl_max=lmx)

        # DFS reference (cached or run once, with its time)
        if name in fixtures and "time" in fixtures[name]:
            dfs_build = fixtures[name]["build"]
            dfs_time = fixtures[name]["time"]
        else:
            if not warmed:
                warm_numba(); warmed = True
            t_dfs = time()
            dfs_build = run_dfs(ings, recipes, skill, itype, tier, lmn, lmx, query)
            dfs_time = time() - t_dfs
            fixtures[name] = {"build": dfs_build, "time": dfs_time,
                              "skill": skill, "item_type": itype}
            with open(FIXTURES, "w", encoding="utf-8") as f:
                json.dump(fixtures, f, indent=2)
        dfs_sc, dfs_v = oracle(dfs_build, rr, tier, query, ibid)

        # separable
        t = time()
        res = solve_separable(query, skill, itype, tier, lmn, lmx,
                              ingredients_raw=ings, recipes=recipes)
        sep_t = time() - t
        sep_build = res["build"]
        sep_sc, sep_v = oracle(sep_build, rr, tier, query, ibid)

        delta = sep_sc - dfs_sc
        ok = sep_v and abs(delta) < 1e-6
        passed += ok; failed += (not ok)
        speedup = (dfs_time / sep_t) if sep_t > 0 else float("inf")

        print("=" * 72)
        print(f"{name}   [{'PASS' if ok else 'FAIL'}{'' if sep_v else ' INVALID'}]   Δscore={delta:+.1f}")
        print(f"  DFS       score={dfs_sc:>11.1f}  time={dfs_time:>7.1f}s  build={dfs_build}")
        print(f"  separable score={sep_sc:>11.1f}  time={sep_t:>7.1f}s  build={sep_build}")
        print(f"  speedup   {speedup:>6.1f}x")
        summary.append((name, dfs_sc, sep_sc, delta, dfs_time, sep_t, speedup, ok, sep_v))

    print("=" * 72)
    print(f"{'case':<14}{'DFS score':>12}{'sep score':>12}{'Δ':>9}{'DFS t':>8}{'sep t':>8}{'speedup':>9}  res")
    print("-" * 72)
    for name, dsc, ssc, d, dt, st, sp, ok, sv in summary:
        print(f"{name:<14}{dsc:>12.1f}{ssc:>12.1f}{d:>9.1f}{dt:>7.1f}s{st:>7.1f}s{sp:>8.1f}x  "
              f"{'PASS' if ok else 'FAIL'}{'' if sv else ' INVALID'}")
    print("-" * 72)
    print(f"{passed} passed, {failed} failed  (fixtures: {FIXTURES})")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
