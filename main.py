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
        "intReq":        {"ingredient_filter": True, "weight": 100000},
        "durability": {"min": 60, "weight": 1},
    }

    skill = "ARMOURING"
    item_type = "CHESTPLATE"
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


























