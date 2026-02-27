import math
from typing import NamedTuple


TIER_MULT = (0.0, 1.0, 1.25, 1.4)


class Recipe(NamedTuple):
    """
    Class used to store all the base stats of a recipe.
    Stores :
        - durability range
        - scaled durability ranged, using material tier multiplier
    """

    base_dura_min: int
    base_dura_max: int
    scaled_dura_min: int
    scaled_dura_max: int


def build_recipe(raw_recipe: dict, tier: int) -> Recipe:
    recipe_data = raw_recipe.data
    
    base_dura_min = recipe_data["durability"]["minimum"]
    base_dura_max = recipe_data["durability"]["maximum"]

    mult = TIER_MULT[tier]

    # Apply material multiplier exactly like craft.js
    scaled_dura_min = math.floor(base_dura_min * mult)
    scaled_dura_max = math.floor(base_dura_max * mult)

    return Recipe(
        base_dura_min=base_dura_min,
        base_dura_max=base_dura_max,
        scaled_dura_min=scaled_dura_min,
        scaled_dura_max=scaled_dura_max,
    )