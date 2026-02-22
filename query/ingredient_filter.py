"""
ingredient_filter.py

Filters raw ingredients according to Query rules.

Exact same logic as original implementation,
adapted to dense stat vectors.

Effectiveness filtering removed.
"""


def filter_raw_ingredients(
    ingredients_raw,
    stat_index,
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
    min_dura = query.min_durability
    item_type = query.item_type

    for ing in ingredients_raw:

        # ---- Item type filter ----
        if item_type is not None:
            skills = ing.get("skills")
            if not skills or item_type not in skills:
                continue

        keep = False

        stats = ing["stats"]

        # ---- Positive Durability filter ----
        if min_dura is not None:
            dura_idx = stat_index.get("durability")
            if dura_idx is not None:
                if stats[dura_idx] > 0:
                    filtered.append(ing)
                    continue

        # ---- Normal stats ----
        for stat_name, idx in stat_index.items():

            # Skip stats not used in query
            if not (has_min[idx] or has_max[idx] or weights[idx] != 0):
                continue

            value = stats[idx]

            # Since rolls are removed:
            min_val = value
            max_val = value

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