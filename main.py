from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB, build_dual_db
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients, split_and_cull_by_sign
from utils.hash_generator import generate_crafter_url
from data.stats import STAT_INDEX, STAT_COUNT, CONSU_SKILLS
from data.meta_set_loader import load_meta_sets

from core.search_engine import search, search_pipelined
from core.warmup import warm_numba
from separable_search import solve_separable, UnsupportedQueryError

from time import time
import cProfile
import pstats

# w7w7w7 Revolution

dps = 497
pctW = 1000
def raw(pct, dps=dps) : return 10*pct/dps 
def is_null(val, epsilon=0.6) : return abs(val) < epsilon

nScale = 396
eScale = 80
tScale = 0
wScale = 0
fScale = 0
aScale = 28

nDmgWeaponMin = 0
nDmgWeaponMax = 0
eDmgWeaponMin = 0
eDmgWeaponMax = 0
tDmgWeaponMin = 0
tDmgWeaponMax = 0
wDmgWeaponMin = 0
wDmgWeaponMax = 0
fDmgWeaponMin = 0
fDmgWeaponMax = 0
aDmgWeaponMin = 209
aDmgWeaponMax = 275

nWeaponAvg = nDmgWeaponMin + ((nDmgWeaponMax - nDmgWeaponMin) / 2)
eWeaponAvg = eDmgWeaponMin + ((eDmgWeaponMax - eDmgWeaponMin) / 2)
tWeaponAvg = tDmgWeaponMin + ((tDmgWeaponMax - tDmgWeaponMin) / 2)
wWeaponAvg = wDmgWeaponMin + ((wDmgWeaponMax - wDmgWeaponMin) / 2)
fWeaponAvg = fDmgWeaponMin + ((fDmgWeaponMax - fDmgWeaponMin) / 2)
aWeaponAvg = aDmgWeaponMin + ((aDmgWeaponMax - aDmgWeaponMin) / 2)

nDmg = nWeaponAvg * nScale
eDmg = eWeaponAvg * nScale + nWeaponAvg * eScale + eWeaponAvg * eScale + tWeaponAvg * eScale + wWeaponAvg * eScale + fWeaponAvg * eScale + aWeaponAvg * eScale
tDmg = tWeaponAvg * nScale + nWeaponAvg * tScale + eWeaponAvg * tScale + tWeaponAvg * tScale + wWeaponAvg * tScale + fWeaponAvg * tScale + aWeaponAvg * tScale
wDmg = wWeaponAvg * nScale + nWeaponAvg * wScale + eWeaponAvg * wScale + tWeaponAvg * wScale + wWeaponAvg * wScale + fWeaponAvg * wScale + aWeaponAvg * wScale
fDmg = fWeaponAvg * nScale + nWeaponAvg * fScale + eWeaponAvg * fScale + tWeaponAvg * fScale + wWeaponAvg * fScale + fWeaponAvg * fScale + aWeaponAvg * fScale
aDmg = aWeaponAvg * nScale + nWeaponAvg * aScale + eWeaponAvg * aScale + tWeaponAvg * aScale + wWeaponAvg * aScale + fWeaponAvg * aScale + aWeaponAvg * aScale

totalDmg = nDmg + eDmg + tDmg + wDmg + fDmg + aDmg
if (totalDmg == 0):
    totalScale = 1
nWeight = pctW * nDmg / totalDmg
eWeight = pctW * eDmg / totalDmg
tWeight = pctW * tDmg / totalDmg
wWeight = pctW * wDmg / totalDmg
fWeight = pctW * fDmg / totalDmg
aWeight = pctW * aDmg / totalDmg

print(aWeight)



def main():

    # ---------- Load all ingredients ----------
    ingredients_raw = load_ingredients("data/ingreds_compress.json")

    # ---------- Build User Query ----------

    user_query = {
        # ===== General Damage =====
        "damPct":   {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "damRaw":   {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},
        "sdPct":    {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "sdRaw":    {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},

        # ===== Neutral =====
        "nDamPct":  {"ingredient_filter": False if is_null(nWeight) else True, "weight": nWeight},
        "nDamRaw":  {"ingredient_filter": False if is_null(nWeight) else True, "weight": raw(nWeight)},

        # ===== Earth =====
        "eDamPct":  {"ingredient_filter": False if is_null(eWeight) else True, "weight": eWeight},
        "eDamRaw":  {"ingredient_filter": False if is_null(eWeight) else True, "weight": raw(eWeight)},
        "eSdPct":   {"ingredient_filter": False if is_null(eWeight) else True, "weight": eWeight},
        "eSdRaw":   {"ingredient_filter": False if is_null(eWeight) else True, "weight": raw(eWeight)},

        # ===== Thunder =====
        "tDamPct":  {"ingredient_filter": False if is_null(tWeight) else True, "weight": tWeight},
        "tDamRaw":  {"ingredient_filter": False if is_null(tWeight) else True, "weight": raw(tWeight)},
        "tSdPct":   {"ingredient_filter": False if is_null(tWeight) else True, "weight": tWeight},
        "tSdRaw":   {"ingredient_filter": False if is_null(tWeight) else True, "weight": raw(tWeight)},

        # ===== Water =====
        "wDamPct":  {"ingredient_filter": False if is_null(wWeight) else True, "weight": wWeight},
        "wDamRaw":  {"ingredient_filter": False if is_null(wWeight) else True, "weight": raw(wWeight)},
        "wSdPct":   {"ingredient_filter": False if is_null(wWeight) else True, "weight": wWeight},
        "wSdRaw":   {"ingredient_filter": False if is_null(wWeight) else True, "weight": raw(wWeight)},

        # ===== Fire =====
        "fDamPct":  {"ingredient_filter": False if is_null(fWeight) else True, "weight": fWeight},
        "fDamRaw":  {"ingredient_filter": False if is_null(fWeight) else True, "weight": raw(fWeight)},
        "fSdPct":   {"ingredient_filter": False if is_null(fWeight) else True, "weight": fWeight},
        "fSdRaw":   {"ingredient_filter": False if is_null(fWeight) else True, "weight": raw(fWeight)},

        # ===== Air =====
        "aDamPct":  {"ingredient_filter": False if is_null(aWeight) else True, "weight": aWeight},
        "aDamRaw":  {"ingredient_filter": False if is_null(aWeight) else True, "weight": raw(aWeight)},
        "aSdPct":   {"ingredient_filter": False if is_null(aWeight) else True, "weight": aWeight},
        "aSdRaw":   {"ingredient_filter": False if is_null(aWeight) else True, "weight": raw(aWeight)},

        # ===== Rainbow ===== // BE CAREFUL IF NEUTRAL WEAPON (RARE)
        "rDamPct":  {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "rDamRaw":  {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},
        "rSdPct":   {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "rSdRaw":   {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},


        # ===== Skill points =====
        "str": {"min": 0, "weight":1.5*pctW, "ingredient_filter":True},
        "dex": {"min": 0, "weight":1.5*pctW, "ingredient_filter":True},
        "int": {"min": 0, "weight":pctW, "ingredient_filter":True},
        "def": {"min": 0, "weight":pctW, "ingredient_filter":True},
        "agi": {"min": 0, "weight":0, "ingredient_filter":False},

        # ===== Sustain =====
        "mr": {"min": 0, "weight":1*pctW},
        #"ms": {"min": 0, "weight":0},
        
        #"hprRaw": {"min": -200, "ingredient_filter":False},
        #"hpBonus": {"min": 2000, "ingredient_filter":False},

        # ===== Requirements =====
        "strReq": {"max": 45, "ingredient_filter":False},
        "dexReq": {"max": 45, "ingredient_filter":False},
        "intReq": {"max": 45, "ingredient_filter":False},
        "defReq": {"max": 15, "ingredient_filter":False},
        "agiReq": {"max": 125, "ingredient_filter":True},

        # ===== Usability =====
        #"spd": {"min": 0, "weight": 0.5},

        "durability": {"min": 75, "weight": 1},
        # "duration": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "charges":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
    }



    

    skill = "ARMOURING"
    dico = {"WEAPONSMITHING":"DAGGER", "WOODWORKING":"RELIK", "TAILORING":"LEGGINGS", "ARMOURING":"CHESTPLATE", "JEWELING":"RING", "COOKING":"FOOD", "SCRIBING":"SCROLL", "ALCHEMISM":"POTION"}
    item_type = dico[skill]

    consumable = skill in CONSU_SKILLS
    
    # ---------- Build Query Object ----------
    query = build_query(
        user_json=user_query,
        search_for_inversion=True, # negative effectiveness included
        item_type=item_type,
        skill=skill,
        consumable=consumable,
        fast_cull=False,  # legacy single-representative pareto cull: unsound under inversion (drops valid candidates) but much faster downstream search. Flip to False for the range-aware sound cull.
        full_meta=False,   # NON OPTIMAL extra pass: complete 6-meta-ingredient builds (extends surviving META_5 rows). See search_engine._search_full_meta.
    )

    # ---------- Load recipes (Materials => Stats) ----------
    recipes = load_recipes("data/recipes_compress.json")

    recipe_raw = find_recipe( # gets recipe stats
        recipes = recipes,
        item_type=item_type,
        skill=skill,
        lvl_min=117,
        lvl_max=119,
    )

    # ---------- Choose solver ----------
    # Try the exact separable solver first (linear queries + supported composites,
    # currently HPR). It raises UnsupportedQueryError for composites it can't handle
    # yet (spell damage / EHP / EHPR), which fall back to the branch-and-bound DFS.
    try:
        res = solve_separable(
            user_query, skill, item_type, tier=3, lvl_min=117, lvl_max=119,
            search_for_inversion=True, consumable=consumable,
            ingredients_raw=ingredients_raw, recipes=recipes,
        )
        best_solution = res["build"]
        tm = res["timings"]
        sc = res["score"]
        print(f"Solver: SEPARABLE (exact)  status={res['status']}"
              + (f"  score={sc:.1f}" if sc is not None else ""))
        print(f"  timing: load {tm['load']:.1f}s + prep {tm['prep']:.2f}s "
              f"+ search {tm['search']:.2f}s")
    except UnsupportedQueryError as ex:
        print(f"Solver: DFS  ({ex})")
        t_warm = time()
        warm_numba()
        print(f"Numba warm-up: {time() - t_warm:.1f}s")

        recipe = build_recipe(recipe_raw, query, tier=3) # builds final recipe using material tier

        # Filter raw ingredients (no cull; the dual-DB split culls per sign).
        filtered_raw = filter_raw_ingredients(ingredients_raw, query, recipe, cull=False)
        pos_list, neg_list = split_and_cull_by_sign(filtered_raw, query, recipe)
        db = build_dual_db(pos_list, neg_list, query)

        print("Raw ingredients:", len(ingredients_raw))
        print("Filtered ingredients:", len(filtered_raw))
        print(f"Search DB: {db.count} (pos {db.pos_count} | neg {db.count - db.pos_count})")

        # search_pipelined prints its own Load/Search/Wall breakdown.
        best_solution = search_pipelined(
            skill, query, recipe, db, max_cull=query.suggested_max_cull,
        )

    print("Best solution:", best_solution)

    if best_solution is not None:

        id_to_name = {int(ing.ing_id): ing.name for ing in ingredients_raw}
        names = [id_to_name[i] for i in best_solution]
        print(names)

        url = generate_crafter_url(
            recipe_id=recipe_raw.data["id"],
            recipe_type=recipe_raw.item_type,
            tier=3,
            ingredient_ids=[int(i) for i in best_solution],
        )
        print("Crafter URL:", url)


profile = False

if __name__ == "__main__":
    start_time = time()
    
    if profile:
        
        profiler = cProfile.Profile()
        profiler.enable()
    
        main()
    
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(30)
        
        """
        lp = LineProfiler()
        lp.add_function(evaluate_leaf)
    
        lp.runctx("main()", globals(), locals())
        lp.print_stats()"""
    
    else:
        main()
    
    print(f"Elapsed time: {time()-start_time:.0f}s")