from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from utils.hash_generator import generate_crafter_url
from data.stats import STAT_INDEX, STAT_COUNT, CONSU_SKILLS
from data.meta_set_loader import load_meta_sets

from core.search_engine import search, search_pipelined
from core.warmup import warm_numba

from time import time
import cProfile
import pstats

# w7w7w7 Revolution

dps = 792
pctW = 1000
def raw(pct, dps=dps) : return 10*pct/dps 
def is_null(val, epsilon=0.6) : return abs(val) < epsilon

nScale = 0.75
eScale = 0
tScale = 0
wScale = 0.25
fScale = 0
aScale = 0

nDmgWeaponMin = 0
nDmgWeaponMax = 0
eDmgWeaponMin = 210
eDmgWeaponMax = 250
tDmgWeaponMin = 0
tDmgWeaponMax = 0
wDmgWeaponMin = 403
wDmgWeaponMax = 585
fDmgWeaponMin = 0
fDmgWeaponMax = 0
aDmgWeaponMin = 160
aDmgWeaponMax = 300

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



def main():

    # ---------- Pre-compile numba kernels ----------
    t_warm = time()
    warm_numba()
    print(f"Numba warm-up: {time() - t_warm:.1f}s")

    # ---------- Load all ingredients ----------
    ingredients_raw = load_ingredients("data/ingreds_compress.json")

    # ---------- Build User Query ----------

    user_query = {
        "strReq": {"max": 0, "ingredient_filter": True},
        "dexReq": {"max": 0, "ingredient_filter": True},
        "intReq": {"max": 0, "ingredient_filter": True},
        "defReq": {"max": 115, "ingredient_filter": True},
        "agiReq": {"max": 85, "ingredient_filter": True},
        "mr":       {"min": 0, "ingredient_filter": True, "weight": 1150},
        "durability": {"min": 100, "weight": 1},
    }
    skill = "JEWELING"
    dico = {"WEAPONSMITHING":"DAGGER", "WOODWORKING":"BOW", "TAILORING":"LEGGINGS", "ARMOURING":"CHESTPLATE", "JEWELING":"RING", "COOKING":"FOOD", "SCRIBING":"SCROLL", "ALCHEMISM":"POTION"}
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
        full_meta=True,   # extra pass: complete 6-meta-ingredient builds (extends surviving META_5 rows). See search_engine._search_full_meta.
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

    recipe = build_recipe(recipe_raw, query, tier=3) # builds final recipe using material tier

    # ---------- Filter raw ingredients ----------
    filtered_raw = filter_raw_ingredients(
        ingredients_raw,
        query,
        recipe,
    )

    # ---------- Build compact DB ----------
    db = IngredientDB(filtered_raw, query)

    print("Raw ingredients:", len(ingredients_raw))
    print("Filtered ingredients:", len(db))
    
    # for ing in filtered_raw:
    #     print(ing.name)
    
    # ---------- Load + Search (pipelined, overlapped) ----------
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