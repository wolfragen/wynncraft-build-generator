"""
main_pareto.py

Entry point for the Pareto-precalc search mode. Generates a frontier of
"promising" recipes for a target item-type/skill/recipe across K user-
defined axes, written to a JSON file the downstream consumer can use.

Run:
    python main_pareto.py
"""

import json
import time

import numpy as np

from data.ingredient_loader import load_ingredients
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes, find_recipe
from data.recipe import build_recipe
from query.query import (
    build_query,
    BUILD_CTX_BASE_REQ_STR, BUILD_CTX_BASE_REQ_DEX,
)
from query.ingredient_filter import filter_raw_ingredients
from data.skillpoint_lookup import SKP_STR, SKP_DEX
from core.warmup import warm_numba

from dsl import LIT, STAT, CTX, ADD, MUL, SP_HEADLINE
from core.pareto_search import run_pareto_search, save_frontier


# ============================================================
# Example axes (the user's reference DPS + mr + hpBonus)
# ============================================================
def build_example_axes():
    """Returns a list of (name, Expr) tuples. Reproduces the user's
    reference axis from the design discussion:
        DPS = (BASE_DPS * (100 + damPct + sdPct) + mdRaw + sdRaw)
              * (1 + headline(60 + base_str_alloc))
              * (1 + headline(40 + base_dex_alloc))
        plus mr and hpBonus as standalone axes.
    Substitution: damRaw is not in the registry, we use mdRaw (melee
    damage raw) as the closest analogue."""
    BASE_DPS = LIT(220.0)
    pct_part = ADD(LIT(100), STAT('damPct'), STAT('sdPct'))
    raw_part = ADD(STAT('mdRaw'), STAT('sdRaw'))
    base = ADD(MUL(BASE_DPS, pct_part), raw_part)
    str_count = ADD(LIT(60), CTX(BUILD_CTX_BASE_REQ_STR))
    dex_count = ADD(LIT(40), CTX(BUILD_CTX_BASE_REQ_DEX))
    dps = MUL(
        base,
        ADD(LIT(1), SP_HEADLINE(SKP_STR, str_count)),
        ADD(LIT(1), SP_HEADLINE(SKP_DEX, dex_count)),
    )
    return [
        ("dps", dps),
        ("mr", STAT('mr')),
        ("hpBonus", STAT('hpBonus')),
    ]


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    warm_numba()
    print(f"Numba warm-up: {time.time() - t0:.1f}s")

    ingredients_raw = load_ingredients("data/ingreds_compress.json")
    recipes = load_recipes("data/recipes_compress.json")

    skill = "ARMOURING"
    item_type = "CHESTPLATE"
    recipe_lvl = (117, 119)
    tier = 3
    threshold = 0.9

    # ----- user-facing query inputs -----
    # (a) stats used to select ingredients (= what the cull/filter sees)
    ingredient_filter_stats = [
        "damPct", "mdRaw", "sdPct", "sdRaw", "mr", "str", "dex", "hpBonus",
    ]
    # (b) hard constraints — recipes failing these are eliminated.
    hard_constraints = {
        "strReq":     {"max": 30},
        "durability": {"min": 100},
    }

    # ----- build the underlying Query -----
    # We use weight=1 on each filter stat to make them active in the
    # Query (active_mask is set by min/max/weight). The weight value
    # itself is not used by the pareto search — only the projection.
    user_query = {}
    for name in ingredient_filter_stats:
        user_query[name] = {"ingredient_filter": True, "weight": 1}
    for name, c in hard_constraints.items():
        # Merge with any pre-existing entry from filter_stats.
        prev = user_query.get(name, {"ingredient_filter": True, "weight": 1})
        user_query[name] = {**prev, **c}

    query = build_query(
        user_json=user_query,
        search_for_inversion=False,
        item_type=item_type,
        skill=skill,
        consumable=False,
    )

    recipe_raw = find_recipe(recipes, item_type=item_type, skill=skill,
                             lvl_min=recipe_lvl[0], lvl_max=recipe_lvl[1])
    recipe = build_recipe(recipe_raw, query, tier=tier)

    filtered = filter_raw_ingredients(ingredients_raw, query, recipe)
    print(f"Raw ingredients: {len(ingredients_raw)}")
    print(f"Filtered ingredients: {len(filtered)}")

    db = IngredientDB(filtered, query)

    axes = build_example_axes()

    result = run_pareto_search(
        skill=skill,
        item_type=item_type,
        recipe_lvl=recipe_lvl,
        tier=tier,
        query=query,
        recipe=recipe,
        db=db,
        axes=axes,
        threshold=threshold,
        out_buffer_size=2_000_000,
    )

    print(f"\n=== Final frontier: {result['axes_ub'].shape[0]} recipes ===")
    print(f"Best per axis: {dict(zip(result['axis_names'], result['best_per_axis']))}")

    # Show the top-3 by each axis as a sanity check.
    if result['axes_ub'].shape[0] > 0:
        for ka, name in enumerate(result['axis_names']):
            order = np.argsort(-result['axes_ub'][:, ka])[:3]
            print(f"\nTop-3 by {name}:")
            for rank, idx in enumerate(order):
                ub = result['axes_ub'][idx]
                lb = result['axes_lb'][idx]
                m_n = int(result['meta_n'][idx])
                m_idx = int(result['meta_index'][idx])
                voids = result['void_choices'][idx]
                print(f"  #{rank+1}: lb={lb} ub={ub} META_{m_n} m={m_idx} voids={voids.tolist()}")

    # ----- save frontier to disk -----
    out_path = "data/precalc/pareto_example.json"
    save_frontier(result, db, out_path)
    print(f"\nFrontier saved to {out_path}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
