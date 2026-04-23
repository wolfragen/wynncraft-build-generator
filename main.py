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



def main():

    # ---------- Pre-compile numba kernels ----------
    t_warm = time()
    warm_numba()
    print(f"Numba warm-up: {time() - t_warm:.1f}s")

    # ---------- Load all ingredients ----------
    ingredients_raw = load_ingredients("data/ingreds_compress.json")

    # ---------- Build User Query ----------
    user_query = {
        "mr": {"min": 1, "weight": 10000, "ingredient_filter":True},
        "hprEff": {"min": 1, "weight": 1000, "ingredient_filter":True},
        "str": {"min": 1, "ingredient_filter":True},
        "dex": {"min": 1, "ingredient_filter":True},
        "int": {"min": 1, "ingredient_filter":True},
        "def": {"min": 1, "ingredient_filter":True},
        "agi": {"min": 1, "ingredient_filter":True},
        "strReq": {"max": 50, "ingredient_filter":True},
        "dexReq": {"max": 100, "ingredient_filter":True},
        "intReq": {"max": 100, "ingredient_filter":True},
        "defReq": {"max": 100, "ingredient_filter":True},
        "agiReq": {"max": 100, "ingredient_filter":True},
        "durability": {"min": 40, "weight": 1},
        #"charges": {"min": 3},
    }

    skill = "WEAPONSMITHING"
    item_type = "SPEAR"
    consumable = skill in CONSU_SKILLS
    
    # ---------- Build Query Object ----------
    query = build_query(
        user_json=user_query,
        search_for_inversion=True, # negative effectiveness included
        item_type=item_type,
        skill=skill,
        consumable=consumable,
    )

    # ---------- Load recipes (Materials => Stats) ----------
    recipes = load_recipes("data/recipes_compress.json")

    recipe_raw = find_recipe( # gets recipe stats
        recipes = recipes,
        item_type=item_type,
        skill=skill,
        lvl_min=103,
        lvl_max=105,
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
    t_pipeline = time()
    best_solution = search_pipelined(
        skill, query, recipe, db, max_cull=query.suggested_max_cull,
    )
    print(f"Load+search pipeline: {time() - t_pipeline:.1f}s")

    print("Best solution:", best_solution)
    
    if best_solution is not None:
        
        id_to_name = {int(ing.ing_id): ing.name for ing in ingredients_raw}
        names = [id_to_name[i] for i in best_solution]
        print(names)

        """
        url = generate_crafter_url(
            recipe_json_id=best_solution,
            tier=3,
            ingredient_json_ids=db.json_ids,
            raw_ingredients=ingredients_raw,
            raw_recipes=recipes_data,
        )

        print("Crafter URL:", url)"""


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
    
"""
TODO : 
derived_type        = [HPR_EFF, DPS]

derived_dep_start   = [0, 2]
derived_dep_count   = [2, 4]

derived_deps        =
[
  idx_hprRaw,
  idx_hprPct,

  idx_sdRaw,
  idx_sdPct,
  idx_str,
  idx_dex
]
"""


























