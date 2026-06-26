"""
validate_solver.py — validate ANY crafting search/solver against the DFS.

Generic harness: pick a solver from the SOLVERS registry (default "separable") and
check, for every case, that it returns a build scoring the SAME as the DFS optimum
through the craft.js oracle (and that the build is VALID). The DFS is the ground
truth; its optima are cached in tests/fixtures/dfs_results.json (with timings) so
future runs validate fast without re-running the slow DFS. Delete that file to
regenerate.

To validate a NEW solver later, register it in SOLVERS (a callable
    (query, skill, item_type, tier, lvl_min, lvl_max, ingredients_raw, recipes)
    -> list[6 ingredient ids in slot order]
) and run:  python tests/validate_solver.py <name>

Queries are FROZEN here (not imported from main_*) so the suite — and the cached
fixtures — stay reproducible regardless of changes elsewhere.

Run from python/:  python tests/validate_solver.py [solver_name]
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

FIXTURES = os.path.join(_HERE, "fixtures", "dfs_results.json")


# ----------------------------------------------------------------------------
# Solvers under test. A solver is a callable returning the 6 build ids (slot
# order). Register new search functions here to validate them with this suite.
# ----------------------------------------------------------------------------
def _solve_separable(query, skill, item_type, tier, lmn, lmx, ings, recipes):
    res = solve_separable(query, skill, item_type, tier, lmn, lmx,
                          consumable=(skill in CONSU_SKILLS),
                          ingredients_raw=ings, recipes=recipes)
    return res["build"]


SOLVERS = {
    "separable": _solve_separable,
}


# ----------------------------------------------------------------------------
# Frozen test query (heavy 38-stat ARMOURING build). Kept byte-stable here so the
# cached DFS fixtures stay valid; do NOT replace with an import from main_*.
# ----------------------------------------------------------------------------
def _arm_full_query():
    pctW = 1000

    def raw(x):
        return 10 * x / 497

    def is_null(v, eps=0.6):
        return abs(v) < eps

    nScale, eScale, tScale, wScale, fScale, aScale = 396, 80, 0, 0, 0, 28
    avg = {k: 0.0 for k in "netwfa"}
    avg["a"] = 209 + (275 - 209) / 2
    nDmg = avg["n"] * nScale
    eDmg = sum(avg[x] * eScale for x in "netwfa") + avg["e"] * nScale
    tDmg = sum(avg[x] * tScale for x in "netwfa") + avg["t"] * nScale
    wDmg = sum(avg[x] * wScale for x in "netwfa") + avg["w"] * nScale
    fDmg = sum(avg[x] * fScale for x in "netwfa") + avg["f"] * nScale
    aDmg = sum(avg[x] * aScale for x in "netwfa") + avg["a"] * nScale
    tot = (nDmg + eDmg + tDmg + wDmg + fDmg + aDmg) or 1
    W = dict(zip("netwfa", (pctW * d / tot for d in (nDmg, eDmg, tDmg, wDmg, fDmg, aDmg))))

    q = {}
    for el in "netwfa":
        ew = W[el]
        q[f"{el}DamPct"] = {"ingredient_filter": not is_null(ew), "weight": ew}
        q[f"{el}DamRaw"] = {"ingredient_filter": not is_null(ew), "weight": raw(ew)}
        q[f"{el}SdPct"] = {"ingredient_filter": not is_null(ew), "weight": ew}
        q[f"{el}SdRaw"] = {"ingredient_filter": not is_null(ew), "weight": raw(ew)}
    q.update({
        "damPct": {"ingredient_filter": True, "weight": pctW},
        "damRaw": {"ingredient_filter": True, "weight": raw(pctW)},
        "sdPct":  {"ingredient_filter": True, "weight": pctW},
        "sdRaw":  {"ingredient_filter": True, "weight": raw(pctW)},
        "rDamPct": {"ingredient_filter": True, "weight": pctW},
        "rDamRaw": {"ingredient_filter": True, "weight": raw(pctW)},
        "rSdPct":  {"ingredient_filter": True, "weight": pctW},
        "rSdRaw":  {"ingredient_filter": True, "weight": raw(pctW)},
        "str": {"min": 0, "weight": 1.5 * pctW, "ingredient_filter": True},
        "dex": {"min": 0, "weight": 1.5 * pctW, "ingredient_filter": True},
        "int": {"min": 0, "weight": pctW, "ingredient_filter": True},
        "def": {"min": 0, "weight": pctW, "ingredient_filter": True},
        "agi": {"min": 0, "weight": 0, "ingredient_filter": False},
        "mr":  {"min": 0, "weight": 1 * pctW},
        "strReq": {"max": 45, "ingredient_filter": False},
        "dexReq": {"max": 45, "ingredient_filter": False},
        "intReq": {"max": 45, "ingredient_filter": False},
        "defReq": {"max": 15, "ingredient_filter": False},
        "agiReq": {"max": 125, "ingredient_filter": True},
        "durability": {"min": 75, "weight": 1},
    })
    return q


# (name, skill, item_type, lvl_min, lvl_max, query) — all composite-free.
CASES = [
    ("arm_sustain", "ARMOURING", "CHESTPLATE", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "hpBonus": {"weight": 50},
      "durability": {"min": 75, "weight": 1}}),
    ("arm_full", "ARMOURING", "CHESTPLATE", 117, 119, _arm_full_query()),
    ("arm_tightreq", "ARMOURING", "CHESTPLATE", 117, 119,
     dict(_arm_full_query(), strReq={"max": 10, "ingredient_filter": False})),
    ("tai_legs", "TAILORING", "LEGGINGS", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 400}, "spd": {"weight": 400},
      "hpBonus": {"weight": 50}, "durability": {"min": 50, "weight": 1}}),
    ("wea_dagger", "WEAPONSMITHING", "DAGGER", 117, 119,
     {"damPct": {"weight": 1000}, "damRaw": {"weight": 200}, "sdPct": {"weight": 1000},
      "sdRaw": {"weight": 200}, "mr": {"weight": 300}, "durability": {"min": 50, "weight": 1}}),
    ("jew_ring", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"weight": 300},
      "hpBonus": {"weight": 50}, "durability": {"min": 30, "weight": 1}}),
    ("arm_tank", "ARMOURING", "CHESTPLATE", 117, 119,
     {"hpBonus": {"weight": 100, "min": 2000}, "hprRaw": {"weight": 50},
      "mr": {"weight": 500}, "durability": {"min": 75, "weight": 1}}),
    ("wea_spell", "WEAPONSMITHING", "DAGGER", 117, 119,
     {"sdPct": {"weight": 1000}, "sdRaw": {"weight": 300}, "int": {"min": 0, "weight": 1500},
      "mr": {"weight": 400}, "intReq": {"max": 45}, "durability": {"min": 50, "weight": 1}}),
    ("jew_minimise", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "agi": {"weight": -800}, "durability": {"min": 30, "weight": 1}}),
    ("arm_maxcon", "ARMOURING", "CHESTPLATE", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"max": 0},
      "durability": {"min": 75, "weight": 1}}),
    ("tai_damage", "TAILORING", "LEGGINGS", 117, 119,
     {"damPct": {"weight": 1000}, "damRaw": {"weight": 200}, "sdPct": {"weight": 1000},
      "sdRaw": {"weight": 200}, "str": {"min": 0, "weight": 1500}, "dex": {"min": 0, "weight": 1500},
      "mr": {"weight": 500}, "strReq": {"max": 45}, "dexReq": {"max": 45},
      "durability": {"min": 50, "weight": 1}}),
    ("jew_full", "JEWELING", "RING", 117, 119,
     {"mr": {"weight": 1000}, "ms": {"weight": 500}, "spd": {"weight": 300},
      "hpBonus": {"weight": 50}, "int": {"min": 0, "weight": 800}, "intReq": {"max": 30},
      "durability": {"min": 30, "weight": 1}}),
]


def run_dfs(ings, recipes, skill, item_type, tier, lmn, lmx, query):
    consumable = skill in CONSU_SKILLS
    q = build_query(user_json=query, search_for_inversion=True, item_type=item_type,
                    skill=skill, consumable=consumable, fast_cull=False, full_meta=False)
    rec = build_recipe(find_recipe(recipes=recipes, item_type=item_type, skill=skill,
                                   lvl_min=lmn, lvl_max=lmx), q, tier=tier)
    filt = filter_raw_ingredients(ings, q, rec, cull=False)
    pos, neg = split_and_cull_by_sign(filt, q, rec)
    ddb = build_dual_db(pos, neg, q)
    best = search_pipelined(skill, q, rec, ddb, max_cull=q.suggested_max_cull)
    return [int(i) for i in best]


def oracle(build, recipe_raw, tier, query, ibid):
    entries = [ibid[i] for i in build]
    eff = compute_effectiveness(entries)
    r = score_query(compute_crafted_stats(recipe_raw.data, tier, entries, eff), query)
    return r["score"], r["valid"]


def main():
    solver_name = sys.argv[1] if len(sys.argv) > 1 else "separable"
    if solver_name not in SOLVERS:
        print(f"unknown solver '{solver_name}'. registered: {list(SOLVERS)}")
        return False
    solver = SOLVERS[solver_name]

    os.makedirs(os.path.dirname(FIXTURES), exist_ok=True)
    fixtures = {}
    if os.path.exists(FIXTURES):
        with open(FIXTURES, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

    ings = load_ingredients("data/ingreds_compress.json")
    _, ibid = _index_ingredients("data/ingreds_compress.json")
    recipes = load_recipes("data/recipes_compress.json")
    warmed = False

    print(f"Validating solver: {solver_name}\n")
    passed = failed = 0
    summary = []
    for name, skill, itype, lmn, lmx, query in CASES:
        tier = 3
        rr = find_recipe(recipes=recipes, item_type=itype, skill=skill, lvl_min=lmn, lvl_max=lmx)

        # DFS ground truth (cached or run once, with its time)
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

        # solver under test (wall-clock timed, generic)
        t = time()
        sol_build = solver(query, skill, itype, tier, lmn, lmx, ings, recipes)
        sol_t = time() - t
        sol_sc, sol_v = oracle(sol_build, rr, tier, query, ibid) if sol_build else (float("-inf"), False)

        delta = sol_sc - dfs_sc
        ok = sol_v and abs(delta) < 1e-6

        # Persist the solver's result alongside the DFS ground truth, so the
        # fixtures hold a comparison record for every registered solver.
        fixtures[name]["dfs_score"] = round(dfs_sc, 1)
        fixtures[name].setdefault("solvers", {})[solver_name] = {
            "build": sol_build, "score": round(sol_sc, 1),
            "time": round(sol_t, 3), "valid": bool(sol_v), "matches_dfs": bool(ok),
        }
        with open(FIXTURES, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, indent=2)

        passed += ok; failed += (not ok)
        speedup = (dfs_time / sol_t) if sol_t > 0 else float("inf")

        print("=" * 72)
        print(f"{name}   [{'PASS' if ok else 'FAIL'}{'' if sol_v else ' INVALID'}]   Δscore={delta:+.1f}")
        print(f"  DFS        score={dfs_sc:>11.1f}  time={dfs_time:>7.1f}s  build={dfs_build}")
        print(f"  {solver_name:<9} score={sol_sc:>11.1f}  time={sol_t:>7.1f}s  build={sol_build}")
        print(f"  speedup    {speedup:>6.1f}x")
        summary.append((name, dfs_sc, sol_sc, delta, dfs_time, sol_t, speedup, ok, sol_v))

    print("=" * 72)
    print(f"{'case':<14}{'DFS score':>12}{'sol score':>12}{'Δ':>9}{'DFS t':>8}{'sol t':>8}{'speedup':>9}  res")
    print("-" * 72)
    for nm, dsc, ssc, d, dt, st, sp, ok, sv in summary:
        print(f"{nm:<14}{dsc:>12.1f}{ssc:>12.1f}{d:>9.1f}{dt:>7.1f}s{st:>7.1f}s{sp:>8.1f}x  "
              f"{'PASS' if ok else 'FAIL'}{'' if sv else ' INVALID'}")
    print("-" * 72)
    print(f"solver={solver_name}: {passed} passed, {failed} failed  (fixtures: {FIXTURES})")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
