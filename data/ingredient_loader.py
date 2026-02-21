import json


def load_ingredients(path: str):
    """
    Load ingredient JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stat_index(ingredients_raw):
    """
    Assign one uuid to each stat.
    
    Returns (stat_index, num_stats)
    """
    keys = set()

    for ing in ingredients_raw:
        ids = ing.get("ids") # Checks for all the stats under the "ids" section of an ingredient
        if ids:
            keys.update(ids.keys()) # Adds the ids that aren't in the "keys" array.

    stat_index = {k: i for i, k in enumerate(sorted(keys))} # Sorts keys and creates final dict
    return stat_index, len(stat_index)

































