"""
craft_core.py

Import-safe home of the craft.js-faithful pure functions, shared by the URL
decoder (`main_decode.py`) and the MILP solver (`milp/`).

These functions mirror wynnbuilder's `craft.js`:
  * compute_effectiveness  — posMods → per-slot effectiveness (% , base 100).
    PURELY ADDITIVE across ingredients, which is what makes the MILP's linear
    efficiency model exact (E[p] = 100 + Σ_q b[q,·,p]).
  * compute_crafted_stats  — final crafted item stats (per stat: min..max),
    with floor/round and recipe-base injection.
  * score_query            — weighted score + VALID/INVALID verdict for a query
    of the SAME shape main.py uses.

This module has NO module-level side effects and does not import the data layer
at import time (`score_query` imports `data.stats` lazily), so importing it is
cheap and safe. It must be imported from a context where the project root
(`python/`) is on sys.path — both `main_decode.py` and `milp`/`main_milp.py`
guarantee that (run from `python/`).
"""

import json
import math


# craft.js:310 — tier multiplier per material tier.
TIER_MULT = (0.0, 1.0, 1.25, 1.4)

ARMOR_TYPES = {"helmet", "chestplate", "leggings", "boots"}
WEAPON_TYPES = {"wand", "spear", "bow", "dagger", "relik"}
ACCESSORY_TYPES = {"ring", "necklace", "bracelet"}
CONSUMABLE_TYPES = {"food", "potion", "scroll"}


# ============================================================
# Raw JSON loaders (id-indexed)
# ============================================================

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_ingredients(path):
    raw = _load_json(path)
    return raw, {int(e["id"]): e for e in raw}


def _index_recipes(path):
    raw = _load_json(path)["recipes"]
    return raw, {int(r["id"]): r for r in raw}


# ============================================================
# Effectiveness matrix (mirrors craft.js:407-455)
# ============================================================

def compute_effectiveness(ingredient_entries):
    """
    eff[i][j] starts at 100. For each ingredient at slot n=2*i+j, walk its
    posMods and apply the wynnbuilder rules:
      - above / under  : whole column j (rows above / below i)
      - left / right   : single neighbor (i, j-1) / (i, j+1)
      - touching       : 4-neighbors (orthogonal)
      - notTouching    : everything not on the 4-neighborhood (i,j included? no)
    Returns flat list of 6 ints (% effectiveness, slot order 0..5).
    """
    eff = [[100, 100], [100, 100], [100, 100]]
    for n, ingred in enumerate(ingredient_entries):
        i, j = n // 2, n % 2
        for key, value in (ingred.get("posMods") or {}).items():
            if not value:
                continue
            if key == "above":
                for k in range(i - 1, -1, -1):
                    eff[k][j] += value
            elif key == "under":
                for k in range(i + 1, 3):
                    eff[k][j] += value
            elif key == "left":
                if j == 1:
                    eff[i][j - 1] += value
            elif key == "right":
                if j == 0:
                    eff[i][j + 1] += value
            elif key == "touching":
                for k in range(3):
                    for l in range(2):
                        if (abs(k - i) == 1 and abs(l - j) == 0) \
                           or (abs(k - i) == 0 and abs(l - j) == 1):
                            eff[k][l] += value
            elif key == "notTouching":
                for k in range(3):
                    for l in range(2):
                        if (abs(k - i) > 1) \
                           or (abs(k - i) == 1 and abs(l - j) == 1):
                            eff[k][l] += value
    return [eff[i][j] for i in range(3) for j in range(2)]


# ============================================================
# Final crafted stats (mirror craft.js:407-510 + recipe.py:84-108)
# ============================================================

def _js_round(x):
    """JS Math.round: half rounds toward +inf, for any sign."""
    return math.floor(x + 0.5)


def compute_crafted_stats(recipe, tier, ingredient_entries, eff_flat,
                          atk_speed="NORMAL"):
    """
    Build the {stat_name -> {min, max}} dict for the crafted item.

    For armor/weapon, the recipe's healthOrDamage is injected the same way
    `data/recipe.py` does for the search engine: it lands in `hpBonus` for
    armor, and in the weapon's intrinsic neutral damage (returned separately
    as `nDamBase`). This way a query written against `main.py` semantics
    sees identical numbers.
    """
    item_type = recipe["type"]
    type_lower = item_type.lower()

    if type_lower in ARMOR_TYPES:
        category = "armor"
    elif type_lower in WEAPON_TYPES:
        category = "weapon"
    elif type_lower in ACCESSORY_TYPES:
        category = "accessory"
    elif type_lower in CONSUMABLE_TYPES:
        category = "consumable"
    else:
        category = "unknown"

    matmult = TIER_MULT[tier]
    hod_low = recipe["healthOrDamage"]["minimum"]
    hod_high = recipe["healthOrDamage"]["maximum"]

    if category == "consumable":
        durability = None
        duration = [_js_round(recipe["duration"]["minimum"] * matmult),
                    _js_round(recipe["duration"]["maximum"] * matmult)]
    else:
        durability = [_js_round(recipe["durability"]["minimum"] * matmult),
                      _js_round(recipe["durability"]["maximum"] * matmult)]
        duration = None

    base_hp = None
    nDamBase = None
    if category == "armor":
        base_hp = (math.floor(hod_low * matmult), math.floor(hod_high * matmult))
    elif category == "weapon":
        ratio = 2.05
        if atk_speed == "SLOW":
            ratio /= 1.5
        elif atk_speed == "NORMAL":
            ratio = 1
        elif atk_speed == "FAST":
            ratio /= 2.5
        nb_low = math.floor(math.floor(hod_low * matmult) * ratio)
        nb_high = math.floor(math.floor(hod_high * matmult) * ratio)
        nDamBase = (nb_low, nb_high)

    stats = {}  # name -> {'min': int, 'max': int}

    def _add(name, lo, hi):
        ent = stats.get(name)
        if ent is None:
            stats[name] = {"min": lo, "max": hi}
        else:
            ent["min"] += lo
            ent["max"] += hi

    for n, ingred in enumerate(ingredient_entries):
        eff_pct = eff_flat[n]
        # JS does (eff/100).toFixed(2); 2-decimal float is exact for integer effs.
        eff_mult = round(eff_pct / 100, 2)
        is_powder = bool(ingred.get("isPowder"))

        # ----- itemIDs: req stats + dura -----
        for key, value in (ingred.get("itemIDs") or {}).items():
            if key == "dura":
                if category == "consumable":
                    continue  # consumables use consumableIDs.dura
                durability[0] += value
                durability[1] += value
                continue
            if category == "consumable":
                continue  # consumables NEVER get reqs (craft.js:466)
            v = value if is_powder else _js_round(value * eff_mult)
            _add(key, v, v)

        # ----- consumableIDs: duration + charges -----
        for key, value in (ingred.get("consumableIDs") or {}).items():
            if key == "dura":
                if category == "consumable":
                    duration[0] += value
                    duration[1] += value
            else:  # 'charges' and friends
                _add(key, value, value)

        # ----- ids: rolled stats -----
        for key, val in (ingred.get("ids") or {}).items():
            if isinstance(val, dict):
                lo = val.get("min", val.get("minimum", 0))
                hi = val.get("max", val.get("maximum", 0))
            else:
                lo = hi = val
            r1 = math.floor(lo * eff_mult)
            r2 = math.floor(hi * eff_mult)
            if r1 > r2:
                r1, r2 = r2, r1  # negative eff swap (craft.js:487 sort)
            _add(key, r1, r2)

    # Inject recipe HP into hpBonus so query semantics match main.py.
    if category == "armor" and base_hp is not None:
        _add("hpBonus", base_hp[0], base_hp[1])

    if durability is not None:
        durability = [max(1, math.floor(v)) for v in durability]
        stats["durability"] = {"min": durability[0], "max": durability[1]}
    if duration is not None:
        duration = [max(10, v) for v in duration]
        stats["duration"] = {"min": duration[0], "max": duration[1]}

    return {
        "category": category,
        "type": item_type,
        "skill": recipe["skill"],
        "lvl_min": recipe["lvl"]["minimum"],
        "lvl_max": recipe["lvl"]["maximum"],
        "tier": tier,
        "matmult": matmult,
        "atk_speed": atk_speed if category == "weapon" else None,
        "hp_recipe": base_hp,
        "nDamBase": nDamBase,
        "stats": stats,
    }


# ============================================================
# Query scoring (Python re-implementation of dfs leaf logic)
# ============================================================
# core/search_engine.py dfs() scores at the leaf:
#   score = sum_s w_s * (max_v * 0.99 + min_v * 0.01)
#         + sum_c w_c * (cmax * 0.99 + cmin * 0.01)
# and rejects when has_min[s] && max_v < min_vals[s] or has_max[s] && min_v > max_vals[s].
# We replicate that here on the decoded build's stats. Spell composites and
# ehp/ehpr need extra build context (skp, atk_spd, weapon_dam) — we list
# them under "notes" and exclude from the score, but report base-stat
# violations normally.

def _eval_composite(formula, stats, deps):
    """Returns (cmin, cmax) or None when the composite isn't supported here."""
    def _gv(name):
        e = stats.get(name)
        if e is None:
            return 0, 0
        return e["min"], e["max"]

    if formula == "mul_div_100":
        amin, amax = _gv(deps[0])
        bmin, bmax = _gv(deps[1])
        return (amin * bmin) // 100, (amax * bmax) // 100

    if formula == "raw_to_pct":
        amin, amax = _gv(deps[0])
        bmin, bmax = _gv(deps[1])

        def rtp(r, d):
            if r > 0:
                return (r * (100 + d)) // 100
            if r < 0:
                v = (r * (100 - d)) // 100
                return v if v < 0 else 0
            return 0

        c1, c2 = rtp(amin, bmin), rtp(amax, bmax)
        if c1 > c2:
            c1, c2 = c2, c1
        return c1, c2

    return None  # ehp / ehpr / spell -> need build_ctx


def score_query(crafted, user_query):
    """
    Returns dict: score (float), valid (bool), violations (list[str]),
    notes (list[str] for skipped composites).
    """
    from data.stats import DERIVED_DEPENDENCIES, DERIVED_FORMULA

    stats = crafted["stats"]

    score = 0.0
    valid = True
    violations = []
    notes = []

    for stat_name, cfg in user_query.items():
        if stat_name == "_context":
            continue

        deps = DERIVED_DEPENDENCIES.get(stat_name)
        if deps is None:
            ent = stats.get(stat_name)
            if ent is None:
                cmin = cmax = 0
            else:
                cmin, cmax = ent["min"], ent["max"]
        else:
            formula = DERIVED_FORMULA.get(stat_name)
            if isinstance(formula, tuple) and formula and formula[0] == "spell":
                notes.append(
                    f"{stat_name}: spell composite needs build_ctx "
                    f"(skp/atk_spd/wd) — skipped."
                )
                continue
            r = _eval_composite(formula, stats, deps)
            if r is None:
                notes.append(
                    f"{stat_name}: composite '{formula}' not implemented "
                    f"in decoder — skipped."
                )
                continue
            cmin, cmax = r

        wmin = cfg.get("min")
        wmax = cfg.get("max")
        weight = cfg.get("weight", 0) or 0

        if wmin is not None and cmax < wmin:
            valid = False
            violations.append(
                f"{stat_name}: max {cmax} < required min {wmin} "
                f"(have {cmin}..{cmax})"
            )
        if wmax is not None and cmin > wmax:
            valid = False
            violations.append(
                f"{stat_name}: min {cmin} > required max {wmax} "
                f"(have {cmin}..{cmax})"
            )
        if weight:
            score += weight * (cmax * 0.99 + cmin * 0.01)

    return {
        "score": score,
        "valid": valid,
        "violations": violations,
        "notes": notes,
    }
