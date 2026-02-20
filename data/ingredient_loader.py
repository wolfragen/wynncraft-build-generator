import json


def load_ingredients(path: str):
    """
    Load ingredient JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stat_index(ingredients_raw):
    """
    Build deterministic stat index mapping.
    Returns (stat_index, num_stats)
    """
    keys = set()

    for ing in ingredients_raw:
        ids = ing.get("ids")
        if ids:
            keys.update(ids.keys())

    stat_index = {k: i for i, k in enumerate(sorted(keys))}
    return stat_index, len(stat_index)