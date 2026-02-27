"""
ingredient_filter.py

Filters raw ingredients according to Query rules.

Exact same logic as original implementation,
adapted to dense stat vectors.

Effectiveness filtering removed.
"""

from data.stats import STAT_INDEX, IDX_DURABILITY
from data.ingredient_loader import SKILL_INDEX


def filter_raw_ingredients(
    ingredients_raw,
    query,
):
    """
    Takes all ingredients and the user Query, then returns only useful ingredients.
    """

    filtered = []

    has_min = query.has_min_mask
    has_max = query.has_max_mask
    weights = query.weights
    search_inv = query.search_for_inversion
    skill = query.skill
    stat_index = STAT_INDEX

    # Pre-resolve skill index if needed
    if skill is not None:
        skill_index = SKILL_INDEX.get(skill)
        if skill_index is None:
            return []
    else:
        skill_index = None

    for ing in ingredients_raw:

        # ---- Skill filter ----
        if skill_index is not None:
            if not ing.skills[skill_index]:
                continue
        
        if any(x != 0 for x in ing.pos_mods):
            continue

        keep = False

        min_stats = ing.stats_min
        max_stats = ing.stats_max

        # ---- Normal stats ----
        for stat_name, idx in stat_index.items():

            if idx == IDX_DURABILITY:
                continue

            # Skip stats not used in query
            if not (has_min[idx] or has_max[idx] or weights[idx] != 0):
                continue

            min_val = min_stats[idx]
            max_val = max_stats[idx]

            # ---- Minimum defined and stat can be positive ----
            if has_min[idx] and max_val > 0:
                keep = True
                break

            # ---- Maximum defined and stat can be negative ----
            if has_max[idx] and min_val < 0:
                keep = True
                break

            # ---- Positive weight defined and can be positive ----
            if weights[idx] > 0 and max_val > 0:
                keep = True
                break

            # ---- Negative weight defined and can be negative ----
            if weights[idx] < 0 and min_val < 0:
                keep = True
                break

            # ---- Inversion if enabled ----
            if search_inv:

                if has_min[idx] and min_val < 0:
                    keep = True
                    break

                if has_max[idx] and max_val > 0:
                    keep = True
                    break

                if weights[idx] > 0 and min_val < 0:
                    keep = True
                    break

                if weights[idx] < 0 and max_val > 0:
                    keep = True
                    break

        if keep:
            filtered.append(ing)

    return filtered