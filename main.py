from data.ingredient_loader import load_ingredients, build_stat_index
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import Recipe
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from utils.hash_generator import generate_crafter_url
from core.search_base import SearchBase

from time import time
import cProfile
import pstats

from line_profiler import LineProfiler
from core.leaf_evaluator import evaluate_leaf


def main():

    # ---------- Load raw data ----------
    ingredients_raw = load_ingredients("data/ingreds_compress.json")

    # ---------- Build stat index ----------
    stat_index, num_stats = build_stat_index(ingredients_raw)

    # ---------- Build Query ----------
    user_query = {
        "gSpd": {"min": 1, "weight": 1.0},      # gathering speed %
        #"gXp": {"weight": 0.35},    # gathering XP
        "durability": {"min": 70, "weight": 0.0001},
    }

    skill = "JEWELING"
    item = "RING"

    # ---------- Load recipes ----------
    recipes_data = load_recipes("data/recipes_compress.json")

    recipe_raw = find_recipe(
        recipes_data,
        item_type=item,
        skill=skill,
        lvl_min=103,
        lvl_max=105,
    )

    recipe = Recipe(recipe_raw, tier=3)

    # ---------- Build Query Object ----------
    query = build_query(
        user_query=user_query,
        stat_index=stat_index,
        search_for_inversion=True,
        algorithm="dfs",
        item_type=skill,
    )

    # ---------- Filter raw ingredients ----------
    filtered_raw = filter_raw_ingredients(
        ingredients_raw,
        stat_index,
        query,
    )

    # ---------- Build compact DB ----------
    db = IngredientDB(filtered_raw, stat_index)

    print("Raw ingredients:", len(ingredients_raw))
    print("Filtered ingredients:", db.n)
    
    start_time = time()

    # ---------- Search ----------
    best_score, best_solution = SearchBase.execute(
        db,
        query,
        recipe,
        max_depth=6,
    )

    print(f"Elapsed time: {time()-start_time:.0f}s")
    print("Best score:", best_score)
    print("Best solution:", best_solution)

    if best_solution is not None:

        real_json_ids = [int(db.ids[i]) for i in best_solution]

        url = generate_crafter_url(
            recipe_json_id=recipe_raw["id"],
            tier=3,
            ingredient_json_ids=real_json_ids,
            raw_ingredients=ingredients_raw,
            raw_recipes=recipes_data,
        )

        print("Crafter URL:", url)
        print("Ingredient JSON IDs:", real_json_ids)


profile = False

if __name__ == "__main__":
    
    if profile:
        """
        profiler = cProfile.Profile()
        profiler.enable()
    
        main()
    
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(30)"""
        
        lp = LineProfiler()
        lp.add_function(evaluate_leaf)
    
        lp.runctx("main()", globals(), locals())
        lp.print_stats()
    
    else:
        main()
    
    
"""
831 => Je Ne Sais Quoi
622 => Obelisk Core
668 => Stolen Pearls
593 => Eye of The Beast
635 => Old Treasure\u058e

[831, 622, 593, 635, 831, 622] => 4% gather speed
"""































