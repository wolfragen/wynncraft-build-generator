"""
ingredient_filter.py

Filters raw ingredients according to Query rules.

Exact same logic as original implementation,
adapted to dense stat vectors.

Effectiveness filtering removed.
"""

from data.stats import STAT_INDEX


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
    #min_dura = query.min_durability
    skill = query.skill
    stat_index = STAT_INDEX

    for ing in ingredients_raw:

        # ---- Item type filter ----
        if skill is not None:
            skills = ing.get("skills")
            if not skills or skill not in skills:
                continue

        keep = False

        min_stats = ing["stats_min"]
        max_stats = ing["stats_max"]

        # ---- Normal stats ----
        for stat_name, idx in stat_index.items():
            
            if(stat_name == "durability"):
                continue

            # Skip stats not used in query
            if not (has_min[idx] or has_max[idx] or weights[idx] != 0):
                continue

            # Since rolls are removed:
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