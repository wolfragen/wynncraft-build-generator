from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from utils.hash_generator import generate_crafter_url
from data.stats import STAT_INDEX, STAT_COUNT
from data.meta_set_loader import load_meta_sets

from core.search_engine import search

from time import time
import cProfile
import pstats




def main():

    # ---------- Load all ingredients ----------
    ingredients_raw = load_ingredients("data/ingreds_compress.json")

    # ---------- Build User Query ----------
    user_query = {
        "mr": {"min": 1, "weight": 10000},
        "spd": {"weight": 5000},
        "durability": {"min": 40, "weight": 1},
    }

    skill = "WEAPONSMITHING"
    item_type = "SPEAR"

    # ---------- Load recipes (Materials => Stats) ----------
    recipes = load_recipes("data/recipes_compress.json")

    recipe_raw = find_recipe( # gets recipe stats
        recipes = recipes,
        item_type=item_type,
        skill=skill,
        lvl_min=103,
        lvl_max=105,
    )

    recipe = build_recipe(recipe_raw, tier=3) # builds final recipe using material tier

    # ---------- Build Query Object ----------
    query = build_query(
        user_json=user_query,
        search_for_inversion=True, # negative effectiveness included
        item_type=item_type,
        skill=skill
    )

    # ---------- Filter raw ingredients ----------
    filtered_raw = filter_raw_ingredients(
        ingredients_raw,
        query,
    )

    # ---------- Build compact DB ----------
    db = IngredientDB(filtered_raw, query)

    print("Raw ingredients:", len(ingredients_raw))
    print("Filtered ingredients:", len(db))
    
    meta_sets = load_meta_sets(skill, query, recipe)
    print("Meta sets loaded")
                
    # ---------- Search ----------
    best_solution = search(meta_sets, db, query)

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
831 => Je Ne Sais Quoi
622 => Obelisk Core
668 => Stolen Pearls
593 => Eye of The Beast
635 => Old Treasure\u058e
318 => Serafite

[831, 622, 593, 635, 831, 622] => 4% gather speed
[831, 622, 593, 318, 831, 622] => 

[569, 391, 756, 756, 756] => Borange fluff, Bob's tear, Amber, Amber , Amber, -1

[756, 442, 756, 442, 273, 614] : 3230067.0
[756, 442, 756, 442, 359, 614] : [756, -1, 756, -1, 359, 614] [-1, 756, -1, 359, 614, 756]

visiblement c'est [756, 756, -1, -1, 614, 359] qui a été sauvegardé (identique)
Vérification dans meta_set_loader : il a été cull

cull par : [-1, 756, -1, 756, 614, 668] correct.

[756, 442, 756, 442, 273, 614] was returned again; let's see how [442, 756, 442, 756, 614, 668] did
"""



























