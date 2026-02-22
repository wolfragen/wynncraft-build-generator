"""
ingredient_loader.py

Loads ingredient data from compressed JSON and converts it into
a dense stat representation.

All stats are mapped into a fixed-size flat vector defined in data.stats.

Design goals:
- No dict-based stat storage at runtime
- Deterministic stat indexing
- Compact int16 storage
- Minimal allocations
"""

import json
import numpy as np

from data.stats import STAT_INDEX, STAT_COUNT


def _build_stat_vector(data: dict) -> np.ndarray:
    """
    Build dense stat vector for a single ingredient.

    All values are stored as int16.
    Unknown stats are ignored.
    """
    stats = np.zeros(STAT_COUNT, dtype=np.int16)

    # ------------------------------------------------------------
    # ids
    # ------------------------------------------------------------
    ids = data.get("ids", {})
    for name, value in ids.items():
        idx = STAT_INDEX.get(name)
        if idx is not None:
            stats[idx] = value

    # ------------------------------------------------------------
    # itemIDs
    # ------------------------------------------------------------
    item_ids = data.get("itemIDs", {})
    for name, value in item_ids.items():

        # durability remap
        if name == "dura":
            idx = STAT_INDEX["durability"] # 2 "dura" in the json...
        else:
            idx = STAT_INDEX.get(name)

        if idx is not None:
            stats[idx] = value

    # ------------------------------------------------------------
    # consumableIDs
    # ------------------------------------------------------------
    consumable_ids = data.get("consumableIDs", {})
    for name, value in consumable_ids.items():

        # duration remap
        if name == "dura":
            idx = STAT_INDEX["duration"] # 2 "dura" in the json...
        else:
            idx = STAT_INDEX.get(name)

        if idx is not None:
            stats[idx] = value

    return stats


def load_ingredients(path: str):
    """
    Load ingredients from compressed JSON file.

    Returns:
        List[dict] where each ingredient contains:
            - id
            - name
            - stats (np.ndarray[int16])
            - skills
            - posMods
            - tier
            - lvl
            - type
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ingredients = []

    for entry in raw:

        ingredient = {
            "id": entry["id"],
            "name": entry["name"],
            "stats": _build_stat_vector(entry),
            "skills": entry.get("skills", {}),
            "posMods": entry.get("posMods", {}),
            "tier": entry.get("tier", 0),
            "lvl": entry.get("lvl", 0),
            "type": entry.get("type", 0),
        }

        ingredients.append(ingredient)

    return ingredients