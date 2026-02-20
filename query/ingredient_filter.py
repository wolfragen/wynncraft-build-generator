def filter_raw_ingredients(
    ingredients_raw,
    stat_index,
    query,
):
    """
    Exact rule-based filtering as specified.
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

        # ---- Durability rule (4) ----
        if min_dura is not None:
            dura = ing.get("itemIDs", {}).get("dura", 0)
            if dura > 0:
                filtered.append(ing)
                continue

        ids = ing.get("ids")
        if ids:

            for stat_name, data in ids.items():

                idx = stat_index.get(stat_name)
                if idx is None:
                    continue

                if isinstance(data, dict):
                    min_val = data.get("min", data.get("minimum", 0))
                    max_val = data.get("max", data.get("maximum", 0))
                else:
                    min_val = data
                    max_val = data

                # ---- Rule 1 ----
                if has_min[idx] and max_val > 0:
                    keep = True
                    break

                # ---- Rule 2 ----
                if has_max[idx] and min_val < 0:
                    keep = True
                    break

                # ---- Rule 5 (weight) ----
                if weights[idx] > 0 and max_val > 0:
                    keep = True
                    break

                if weights[idx] < 0 and min_val < 0:
                    keep = True
                    break

                # ---- Rule 3 (inversion) ----
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

        # ---- Effectiveness rule (6, 7) ----
        if not keep:
            pos = ing.get("posMods", {})

            if pos:
                has_pos_eff = False
                has_neg_eff = False

                for v in pos.values():
                    if v > 0:
                        has_pos_eff = True
                    elif v < 0:
                        has_neg_eff = True

                if has_pos_eff:
                    keep = True
                elif search_inv and has_neg_eff:
                    keep = True

        if keep:
            filtered.append(ing)

    return filtered