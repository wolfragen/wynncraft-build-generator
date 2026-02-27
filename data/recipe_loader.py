import json
from typing import NamedTuple

from data.ingredient_loader import SKILL_INDEX


class RawRecipe(NamedTuple):
    item_type: int
    skill_index: int
    lvl_min: int
    lvl_max: int
    data: dict


def load_recipes(path):
    """
    Load recipe JSON file.
    """
    raw = json.load(open(path))

    recipes = []

    for r in raw["recipes"]:
        skill_name = r["skill"]
        skill_index = SKILL_INDEX.get(skill_name, -1)

        recipe = RawRecipe(
            item_type=r["type"],
            skill_index=skill_index,
            lvl_min=r["lvl"]["minimum"],
            lvl_max=r["lvl"]["maximum"],
            data=r,
        )

        recipes.append(recipe)

    return recipes


def find_recipe(recipes, item_type, skill, lvl_min, lvl_max):
    """
    Returns the stats of a specific recipe (using level range, item_type and skill used)
    """

    skill_index = SKILL_INDEX.get(skill)
    if skill_index is None:
        raise ValueError(f"Unknown skill: {skill}")

    for r in recipes:
        if (
            r.item_type == item_type
            and r.skill_index == skill_index
            and r.lvl_min == lvl_min
            and r.lvl_max == lvl_max
        ):
            return r

    raise ValueError("Recipe not found")