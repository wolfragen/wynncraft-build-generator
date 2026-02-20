import json


def load_recipes(path):
    data = json.load(open(path))
    return data["recipes"]  # <-- THIS is the fix


def find_recipe(recipes, item_type, skill, lvl_min, lvl_max):

    for r in recipes:
        if (
            r["type"] == item_type
            and r["skill"] == skill
            and r["lvl"]["minimum"] == lvl_min
            and r["lvl"]["maximum"] == lvl_max
        ):
            return r

    raise ValueError("Recipe not found")