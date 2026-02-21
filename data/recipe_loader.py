import json


def load_recipes(path):
    """
    Load recipe JSON file.
    """
    data = json.load(open(path))
    return data["recipes"]


def find_recipe(recipes, item_type, skill, lvl_min, lvl_max):
    """
    Returns the stats of a specific recipe (using level range, item_type and skill used)
    """

    for r in recipes:
        if (
            r["type"] == item_type
            and r["skill"] == skill
            and r["lvl"]["minimum"] == lvl_min
            and r["lvl"]["maximum"] == lvl_max
        ):
            return r

    raise ValueError("Recipe not found")