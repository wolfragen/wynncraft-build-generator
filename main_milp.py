"""
main_milp.py — MILP entry point.

Mirrors main.py (load → build_query → recipe → candidates → solve → URL) but uses
the exact CP-SAT solver in milp/ instead of the heuristic DFS. The user_query is
the SAME shape main.py uses: any composite-free query pastes in verbatim and is
solvable by main.py too (copy/paste parity, milp/README.md §2). Composite queries
are rejected up front.
"""

from time import time

from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from utils.hash_generator import generate_crafter_url
from data.stats import CONSU_SKILLS

from craft_core import _index_ingredients
from milp.efficiency import build_b_tensor
from milp.model import build_model
from milp.solve import solve_model
from milp.result import verify_milp_pick, print_report


class UnsupportedQueryError(Exception):
    """Raised when a query uses features the v1 MILP does not support (composites)."""


# ---- damage-weight preamble (identical to main.py so the query is identical) ----
dps = 497
pctW = 1000
def raw(pct, dps=dps): return 10 * pct / dps
def is_null(val, epsilon=0.6): return abs(val) < epsilon

nScale, eScale, tScale, wScale, fScale, aScale = 396, 80, 0, 0, 0, 28
aDmgWeaponMin, aDmgWeaponMax = 209, 275
_avg = {k: 0.0 for k in "netwfa"}
_avg["a"] = aDmgWeaponMin + (aDmgWeaponMax - aDmgWeaponMin) / 2

nDmg = _avg["n"] * nScale
eDmg = sum(_avg[x] * eScale for x in "netwfa") + _avg["e"] * nScale
tDmg = sum(_avg[x] * tScale for x in "netwfa") + _avg["t"] * nScale
wDmg = sum(_avg[x] * wScale for x in "netwfa") + _avg["w"] * nScale
fDmg = sum(_avg[x] * fScale for x in "netwfa") + _avg["f"] * nScale
aDmg = sum(_avg[x] * aScale for x in "netwfa") + _avg["a"] * nScale
totalDmg = nDmg + eDmg + tDmg + wDmg + fDmg + aDmg or 1
nWeight = pctW * nDmg / totalDmg
eWeight = pctW * eDmg / totalDmg
tWeight = pctW * tDmg / totalDmg
wWeight = pctW * wDmg / totalDmg
fWeight = pctW * fDmg / totalDmg
aWeight = pctW * aDmg / totalDmg


def build_user_query():
    """Same shape and values as main.py's user_query (composite-free)."""
    return {
        "damPct": {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "damRaw": {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
        "sdPct":  {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "sdRaw":  {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
        "nDamPct": {"ingredient_filter": not is_null(nWeight), "weight": nWeight},
        "nDamRaw": {"ingredient_filter": not is_null(nWeight), "weight": raw(nWeight)},
        "eDamPct": {"ingredient_filter": not is_null(eWeight), "weight": eWeight},
        "eDamRaw": {"ingredient_filter": not is_null(eWeight), "weight": raw(eWeight)},
        "eSdPct":  {"ingredient_filter": not is_null(eWeight), "weight": eWeight},
        "eSdRaw":  {"ingredient_filter": not is_null(eWeight), "weight": raw(eWeight)},
        "tDamPct": {"ingredient_filter": not is_null(tWeight), "weight": tWeight},
        "tDamRaw": {"ingredient_filter": not is_null(tWeight), "weight": raw(tWeight)},
        "tSdPct":  {"ingredient_filter": not is_null(tWeight), "weight": tWeight},
        "tSdRaw":  {"ingredient_filter": not is_null(tWeight), "weight": raw(tWeight)},
        "wDamPct": {"ingredient_filter": not is_null(wWeight), "weight": wWeight},
        "wDamRaw": {"ingredient_filter": not is_null(wWeight), "weight": raw(wWeight)},
        "wSdPct":  {"ingredient_filter": not is_null(wWeight), "weight": wWeight},
        "wSdRaw":  {"ingredient_filter": not is_null(wWeight), "weight": raw(wWeight)},
        "fDamPct": {"ingredient_filter": not is_null(fWeight), "weight": fWeight},
        "fDamRaw": {"ingredient_filter": not is_null(fWeight), "weight": raw(fWeight)},
        "fSdPct":  {"ingredient_filter": not is_null(fWeight), "weight": fWeight},
        "fSdRaw":  {"ingredient_filter": not is_null(fWeight), "weight": raw(fWeight)},
        "aDamPct": {"ingredient_filter": not is_null(aWeight), "weight": aWeight},
        "aDamRaw": {"ingredient_filter": not is_null(aWeight), "weight": raw(aWeight)},
        "aSdPct":  {"ingredient_filter": not is_null(aWeight), "weight": aWeight},
        "aSdRaw":  {"ingredient_filter": not is_null(aWeight), "weight": raw(aWeight)},
        "rDamPct": {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "rDamRaw": {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
        "rSdPct":  {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "rSdRaw":  {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
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
    }


# ---- recipe/profession config (same defaults as main.py) ----
SKILL = "ARMOURING"
ITEM_TYPE = {"WEAPONSMITHING": "DAGGER", "WOODWORKING": "RELIK", "TAILORING": "LEGGINGS",
             "ARMOURING": "CHESTPLATE", "JEWELING": "RING", "COOKING": "FOOD",
             "SCRIBING": "SCROLL", "ALCHEMISM": "POTION"}[SKILL]
TIER = 3
LVL_MIN, LVL_MAX = 117, 119
SEARCH_FOR_INVERSION = True
MAX_TIME_S = 120.0


def main(user_query=None):
    user_query = build_user_query() if user_query is None else user_query
    consumable = SKILL in CONSU_SKILLS

    ingredients_raw = load_ingredients("data/ingreds_compress.json")
    _, ing_by_id = _index_ingredients("data/ingreds_compress.json")

    query = build_query(
        user_json=user_query,
        search_for_inversion=SEARCH_FOR_INVERSION,
        item_type=ITEM_TYPE,
        skill=SKILL,
        consumable=consumable,
    )

    # The single point of divergence from main.py: reject composites up front.
    if query.comp_count > 0:
        raise UnsupportedQueryError(
            f"MILP track does not support composite stats yet "
            f"(query defines {query.comp_count}). Use main.py for this query."
        )

    recipes = load_recipes("data/recipes_compress.json")
    recipe_raw = find_recipe(recipes=recipes, item_type=ITEM_TYPE, skill=SKILL,
                             lvl_min=LVL_MIN, lvl_max=LVL_MAX)
    recipe = build_recipe(recipe_raw, query, tier=TIER)

    # Take EVERY legal ingredient (meta included) — required for optimality.
    candidates = filter_raw_ingredients(
        ingredients_raw, query, recipe,
        include_meta=True, recipe_lvl=recipe_raw.lvl_max,
    )
    db = IngredientDB(candidates, query)
    print(f"Candidates (incl. meta): {db.count}   active stats: {query.stat_count}")

    # b-tensor from the SAME json ids the DB uses (post-sort), via the oracle probe.
    b, E_min, E_max = build_b_tensor(db.json_ids, ing_by_id, query.search_for_inversion)

    t0 = time()
    mm = build_model(db, b, E_min, E_max, query, recipe)
    t_build = time() - t0
    print(f"Model built in {t_build:.2f}s  (W={mm.weight_scale}, BIG={mm.big})")

    res = solve_model(mm, max_time_s=MAX_TIME_S, workers=8)
    solve_t = time() - t0

    if res["chosen_rows"] is None:
        print(f"\nMILP status: {res['status_name']} — no craft satisfies the "
              f"requested constraints.")
        return None

    ingredient_ids = [int(db.json_ids[r]) for r in res["chosen_rows"]]
    url = generate_crafter_url(
        recipe_id=recipe_raw.data["id"],
        recipe_type=recipe_raw.item_type,
        tier=TIER,
        ingredient_ids=ingredient_ids,
    )

    report = verify_milp_pick(ingredient_ids, recipe_raw, TIER, ing_by_id,
                              user_query, mm, res)
    print_report(res, ingredient_ids, url, report, ing_by_id, solve_t=solve_t)
    return {"ids": ingredient_ids, "url": url, "report": report, "res": res}


if __name__ == "__main__":
    start = time()
    main()
    print(f"Total elapsed: {time() - start:.1f}s")
