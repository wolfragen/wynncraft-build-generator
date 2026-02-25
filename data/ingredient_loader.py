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


def _build_stat_vectors(data: dict):
    """
    Build dense min/max stat vectors for a single ingredient.

    Returns:
        stats_min: int16 [STAT_COUNT]
        stats_max: int16 [STAT_COUNT]
    """

    stats_min = np.zeros(STAT_COUNT, dtype=np.int16)
    stats_max = np.zeros(STAT_COUNT, dtype=np.int16)

    # ------------------------------------------------------------
    # ids
    # ------------------------------------------------------------
    ids = data.get("ids", {})
    for name, value in ids.items():

        idx = STAT_INDEX.get(name)
        if idx is None:
            continue

        if isinstance(value, dict):
            min_val = value.get("min", value.get("minimum", 0))
            max_val = value.get("max", value.get("maximum", 0))
        else:
            min_val = value
            max_val = value

        stats_min[idx] = min_val
        stats_max[idx] = max_val

    # ------------------------------------------------------------
    # itemIDs
    # ------------------------------------------------------------
    item_ids = data.get("itemIDs", {})
    for name, value in item_ids.items():

        if name == "dura":
            idx = STAT_INDEX["durability"]
        else:
            idx = STAT_INDEX.get(name)

        if idx is None:
            continue

        if isinstance(value, dict):
            min_val = value.get("min", value.get("minimum", 0))
            max_val = value.get("max", value.get("maximum", 0))
        else:
            min_val = value
            max_val = value

        stats_min[idx] = min_val
        stats_max[idx] = max_val

    # ------------------------------------------------------------
    # consumableIDs
    # ------------------------------------------------------------
    consumable_ids = data.get("consumableIDs", {})
    for name, value in consumable_ids.items():

        if name == "dura":
            idx = STAT_INDEX["duration"]
        else:
            idx = STAT_INDEX.get(name)

        if idx is None:
            continue

        if isinstance(value, dict):
            min_val = value.get("min", value.get("minimum", 0))
            max_val = value.get("max", value.get("maximum", 0))
        else:
            min_val = value
            max_val = value

        stats_min[idx] = min_val
        stats_max[idx] = max_val

    return stats_min, stats_max


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
        stats_min, stats_max = _build_stat_vectors(entry)

        ingredient = {
            "id": entry["id"],
            "name": entry["name"],
            "stats_min": stats_min,
            "stats_max": stats_max,
            "skills": entry.get("skills", {}),
            "posMods": entry.get("posMods", {}),
            "tier": entry.get("tier", 0),
            "lvl": entry.get("lvl", 0),
            "type": entry.get("type", 0),
        }

        ingredients.append(ingredient)

    return ingredients













