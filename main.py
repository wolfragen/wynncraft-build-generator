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
    pctW = 1000
    rawW = pctW/10.8
    user_query = {
        #"mage_meteor": {"weight": 100000, "ingredient_filter": True},
        # ===== Skill Points =====
        # "str": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        "dex": {"ingredient_filter": True, "weight": pctW+250},
        # "int": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "def": {"min": 20, "ingredient_filter": True, "weight": 500},
        # "agi": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Skill Point Requirements =====
        "strReq": {"max": 100, "ingredient_filter": True},
        "dexReq": {"max": 21, "ingredient_filter": True},
        "intReq": {"max": 60, "ingredient_filter": True},
        "defReq": {"max": 25, "ingredient_filter": True},
        "agiReq": {"max": 0, "ingredient_filter": True},

        # ===== Health / Mana / Regen =====
        "hpBonus":  {"min": 4000, "ingredient_filter": False, "weight": 0},
        "hprRaw":   {"min": 235, "ingredient_filter": True, "weight": 0},
        # "hprPct":   {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        "mr":       {"min": -1, "ingredient_filter": True, "weight": 0},
        # "ms":       {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "maxMana":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "ls":       {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "healPct":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Derived (Composite) =====
        # "ehp":  {"min": 0, "weight": 50, "ingredient_filter": True},
        # "ehpr": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "hpr":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== General Damage =====
        "damPct":  {"ingredient_filter": True, "weight": pctW},
        "damRaw":  {"ingredient_filter": True, "weight": rawW},
        #"mdPct":   {"min": 20, "ingredient_filter": True, "weight": 1000},
        # "mdRaw":   {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        "sdPct":   {"ingredient_filter": True, "weight": pctW},
        "sdRaw":   {"ingredient_filter": True, "weight": rawW},
        # "nDamRaw": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "nMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "atkTier": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Neutral =====
        # "nDamPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Earth =====
        "eDamPct": {"ingredient_filter": True, "weight": pctW},
        "eDamRaw": {"ingredient_filter": True, "weight": rawW},
        # "eMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        "eSdPct":  {"ingredient_filter": True, "weight": pctW},
        "eSdRaw":  {"ingredient_filter": True, "weight": rawW},
        # "eDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "eSteal":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Thunder =====
        # "tDamPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "tDamRaw": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "tMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "tSdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "tDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Water =====
        # "wDamPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "wDamRaw": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "wMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "wSdPct":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "wSdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "wDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Fire =====
        #"fDamPct": {"min": 20, "ingredient_filter": True, "weight": 1000},
        # "fDamRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "fMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "fSdPct":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "fSdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "fDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Air =====
        # "aDamPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "aDamRaw": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "aMdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "aSdPct":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "aSdRaw":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "aDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Rainbow =====
        # "rDamPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "rSdPct":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "rDefPct": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Misc Damage / Defense =====
        # "poison": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "thorns": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "ref":    {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Movement =====
        # "spd":       {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "jh":        {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "sprint":    {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "sprintReg": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Misc =====
        # "lb":   {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "lq":   {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "xpb":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "gSpd": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "gXp":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "expd": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "kb":   {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},

        # ===== Special =====
        "durability": {"min": 160, "weight": 1},
        # "duration": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "charges":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
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


























