"""
main_separable.py — entry point for the separable (exact, fast) solver.

Mirrors main.py's query but solves with separable_search instead of the DFS. The
DFS (main.py / core/search_engine.py) is left untouched. Composite queries are
rejected (use main.py for those).
"""

from time import time

from data.stats import CONSU_SKILLS
from utils.hash_generator import generate_crafter_url
from separable_search import solve_separable, UnsupportedQueryError


# ---- damage-weight preamble (identical to main.py so the query is identical) ----
dps = 497
pctW = 1000
def raw(pct, dps=dps): return 10 * pct / dps
def is_null(val, epsilon=0.6): return abs(val) < epsilon

nScale, eScale, tScale, wScale, fScale, aScale = 396, 80, 0, 0, 0, 28
_avg = {k: 0.0 for k in "netwfa"}
_avg["a"] = 209 + (275 - 209) / 2
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
    e = {}
    for el, ew in (("n", nWeight), ("e", eWeight), ("t", tWeight),
                   ("w", wWeight), ("f", fWeight), ("a", aWeight)):
        e[f"{el}DamPct"] = {"ingredient_filter": not is_null(ew), "weight": ew}
        e[f"{el}DamRaw"] = {"ingredient_filter": not is_null(ew), "weight": raw(ew)}
        e[f"{el}SdPct"] = {"ingredient_filter": not is_null(ew), "weight": ew}
        e[f"{el}SdRaw"] = {"ingredient_filter": not is_null(ew), "weight": raw(ew)}
    q = {
        "damPct": {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "damRaw": {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
        "sdPct":  {"ingredient_filter": not is_null(pctW), "weight": pctW},
        "sdRaw":  {"ingredient_filter": not is_null(pctW), "weight": raw(pctW)},
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
    q.update(e)
    return q


SKILL = "ARMOURING"
ITEM_TYPE = {"WEAPONSMITHING": "DAGGER", "WOODWORKING": "RELIK", "TAILORING": "LEGGINGS",
             "ARMOURING": "CHESTPLATE", "JEWELING": "RING", "COOKING": "FOOD",
             "SCRIBING": "SCROLL", "ALCHEMISM": "POTION"}[SKILL]
TIER = 3
LVL_MIN, LVL_MAX = 117, 119


def main(user_query=None):
    user_query = build_user_query() if user_query is None else user_query
    consumable = SKILL in CONSU_SKILLS
    t0 = time()
    try:
        res = solve_separable(user_query, SKILL, ITEM_TYPE, TIER, LVL_MIN, LVL_MAX,
                              consumable=consumable, verbose=True)
    except UnsupportedQueryError as ex:
        print(f"Unsupported: {ex}")
        return None
    if res["build"] is None:
        print("No craft satisfies the requested constraints (INFEASIBLE).")
        return None

    url = generate_crafter_url(
        recipe_id=res["recipe_raw"].data["id"],
        recipe_type=res["recipe_raw"].item_type,
        tier=TIER,
        ingredient_ids=res["build"],
    )
    tm = res["timings"]
    print(f"Status     : {res['status']}  (exact, separable)")
    print(f"Build      : {res['build']}  (meta_n={res['meta_n']})")
    print(f"Score      : {res['score']:.1f}")
    print(f"Timing     : load {tm['load']:.1f}s + prep {tm['prep']:.2f}s + search {tm['search']:.2f}s "
          f"= {time() - t0:.1f}s total")
    print(f"Crafter URL: {url}")
    return res


if __name__ == "__main__":
    main()
