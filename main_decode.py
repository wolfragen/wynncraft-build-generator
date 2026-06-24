"""
main_decode.py

Decode a Wynncraft crafted-item URL produced by wynnbuilder's crafter
(e.g. https://wynnbuilder-beta.github.io/crafter/#4OayaWn0n8Q8t4e81)
and report:

  * recipe (name, type, skill, lvl range), tier and atk speed
  * each ingredient: name, json id, raw min/max stats, posMods
  * the 6 effectiveness slots (per-cell %, after posMods)
  * the final crafted item stats (per stat: min .. max),
    matching wynnbuilder's craft.js semantics:
      - rolled `ids`        : floor(value * eff_mult), rolls re-sorted
      - point `itemIDs`     : round(value * eff_mult)  (req stats)
      - durability/duration : tier-scaled recipe base + sum(value), no eff
      - armor recipe HP     : injected into hpBonus (project convention,
                              matches data/recipe.py:99-100)
  * if a query (same shape as `main.py`) is supplied: the weighted score
    (sum_s w_s * (max*0.99 + min*0.01)) plus a VALID / INVALID verdict and
    the list of constraint violations. Composites that need build context
    (spells, ehp/ehpr) are listed under "notes" and excluded from the score.

Run from the `python/` directory so the relative data paths resolve.
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils.hash_generator import (
    CHARSET,
    ATTACK_SPEED_MAP,
    ATTACK_SPEED_BITLEN,
    _CRAFTED_ENCODING_VERSION,
    _VERSION_BITLEN,
    _ING_ID_BITLEN,
    _RECIPE_ID_BITLEN,
    _NUM_INGS,
    _NUM_MATS,
    _MAT_TIER_BITLEN,
)


CHARSET_INDEX = {c: i for i, c in enumerate(CHARSET)}
ATTACK_SPEED_NAME = {v: k for k, v in ATTACK_SPEED_MAP.items()}

# craft.js:310 — tier multiplier per material tier.
TIER_MULT = (0.0, 1.0, 1.25, 1.4)

ARMOR_TYPES = {"helmet", "chestplate", "leggings", "boots"}
WEAPON_TYPES = {"wand", "spear", "bow", "dagger", "relik"}
ACCESSORY_TYPES = {"ring", "necklace", "bracelet"}
CONSUMABLE_TYPES = {"food", "potion", "scroll"}


# ============================================================
# Bit-level decoder (mirror of EncodingBitVector.append)
# ============================================================

class _BitCursor:
    __slots__ = ("value", "length", "pos")

    def __init__(self, hash_str):
        v = 0
        for i, ch in enumerate(hash_str):
            try:
                v |= CHARSET_INDEX[ch] << (6 * i)
            except KeyError as e:
                raise ValueError(f"Invalid hash char {ch!r}") from e
        self.value = v
        self.length = 6 * len(hash_str)
        self.pos = 0

    def advance(self, n):
        out = (self.value >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return out


def _strip_url(url):
    """Pull out the b64 hash portion of a crafter URL."""
    url = url.strip()
    if "#" in url:
        url = url.split("#", 1)[1]
    elif "/" in url:
        url = url.rsplit("/", 1)[1]
    return url


def decode_crafter_url(url):
    """
    Returns dict: hash, recipe_id, ingredient_ids (6), mat_tiers (2), tier (=mat_tiers[0]).
    Atk speed is consumed later once the recipe type is known.
    """
    hash_str = _strip_url(url)
    cur = _BitCursor(hash_str)

    legacy = cur.advance(1)
    if legacy:
        raise ValueError("Legacy crafter encodings (CR-…) are not supported.")
    version = cur.advance(_VERSION_BITLEN)
    if version != _CRAFTED_ENCODING_VERSION:
        raise ValueError(f"Unsupported crafter encoding version {version}.")

    ing_ids = [cur.advance(_ING_ID_BITLEN) for _ in range(_NUM_INGS)]
    recipe_id = cur.advance(_RECIPE_ID_BITLEN)
    mat_tiers = [cur.advance(_MAT_TIER_BITLEN) + 1 for _ in range(_NUM_MATS)]

    return {
        "hash": hash_str,
        "recipe_id": recipe_id,
        "ingredient_ids": ing_ids,
        "mat_tiers": mat_tiers,
        "tier": mat_tiers[0],
        "_cursor": cur,
    }


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


# ============================================================
# Pretty print
# ============================================================

def _fmt(lo, hi):
    return str(lo) if lo == hi else f"{lo} .. {hi}"


def print_decoded(decoded, recipe, ingreds, eff_flat, crafted, query_result=None):
    print("=" * 64)
    print(f"URL hash         : {decoded['hash']}")
    print(f"Recipe id        : {decoded['recipe_id']}  ({recipe['name']})")
    print(f"Recipe type      : {recipe['type']}   skill: {recipe['skill']}")
    print(f"Recipe lvl       : {recipe['lvl']['minimum']}-{recipe['lvl']['maximum']}")
    print(f"Material tiers   : {decoded['mat_tiers']}  (matmult={crafted['matmult']})")
    if crafted["category"] == "weapon":
        print(f"Attack speed     : {crafted['atk_speed']}")
        print(f"nDamBase         : {crafted['nDamBase'][0]}-{crafted['nDamBase'][1]}")
    if crafted["category"] == "armor":
        lo, hi = crafted["hp_recipe"]
        print(f"Recipe HP        : {_fmt(lo, hi)}  (folded into hpBonus below)")
    print()

    print("Effectiveness slots [row][col]:")
    for i in range(3):
        a = eff_flat[2 * i]
        b = eff_flat[2 * i + 1]
        print(f"  row {i}: [{a:>5}%, {b:>5}%]")
    print()

    print("Ingredients:")
    for n, ing in enumerate(ingreds):
        i, j = n // 2, n % 2
        powder = " (powder)" if ing.get("isPowder") else ""
        print(f"  slot {n}  (i={i}, j={j})  eff={eff_flat[n]:>4}%  "
              f"id={ing['id']}  '{ing['name']}'{powder}  "
              f"tier={ing.get('tier','?')}  lvl={ing.get('lvl','?')}")
        ids = ing.get("ids") or {}
        for k in sorted(ids):
            v = ids[k]
            if isinstance(v, dict):
                lo = v.get("min", v.get("minimum", 0))
                hi = v.get("max", v.get("maximum", 0))
                print(f"      ids.{k:<10s}: {_fmt(lo, hi)}")
            else:
                print(f"      ids.{k:<10s}: {v}")
        item_ids = ing.get("itemIDs") or {}
        for k in sorted(item_ids):
            v = item_ids[k]
            if v != 0 or k == "dura":
                print(f"      itemIDs.{k:<8s}: {v}")
        consu = ing.get("consumableIDs") or {}
        for k in sorted(consu):
            v = consu[k]
            if v != 0 or k in ("dura", "charges"):
                print(f"      consu.{k:<10s}: {v}")
        nz = {k: v for k, v in (ing.get("posMods") or {}).items() if v}
        if nz:
            print(f"      posMods       : {nz}")
    print()

    print(f"Final crafted stats ({len(crafted['stats'])} entries):")
    for k in sorted(crafted["stats"]):
        e = crafted["stats"][k]
        print(f"  {k:<14s}: {_fmt(e['min'], e['max'])}")
    print()

    if query_result is not None:
        status = "VALID" if query_result["valid"] else "INVALID"
        print(f"Query score      : {query_result['score']:.2f}   [{status}]")
        if query_result["violations"]:
            print("Violations:")
            for v in query_result["violations"]:
                print(f"  - {v}")
        if query_result["notes"]:
            print("Notes:")
            for nn in query_result["notes"]:
                print(f"  - {nn}")
        print()


# ============================================================
# Entry point
# ============================================================

def decode_and_report(url, user_query=None,
                      ingred_path="data/ingreds_compress.json",
                      recipe_path="data/recipes_compress.json"):
    decoded = decode_crafter_url(url)

    _, ing_by_id = _index_ingredients(ingred_path)
    _, rec_by_id = _index_recipes(recipe_path)

    recipe = rec_by_id.get(int(decoded["recipe_id"]))
    if recipe is None:
        raise ValueError(f"Recipe id {decoded['recipe_id']} not found.")

    type_lower = recipe["type"].lower()

    cur = decoded["_cursor"]
    if type_lower in WEAPON_TYPES:
        atk_idx = cur.advance(ATTACK_SPEED_BITLEN)
        atk_speed = ATTACK_SPEED_NAME.get(atk_idx, "SLOW")
    else:
        atk_speed = "NORMAL"

    ingreds = []
    for ing_id in decoded["ingredient_ids"]:
        ing = ing_by_id.get(int(ing_id))
        if ing is None:
            raise ValueError(f"Ingredient id {ing_id} not found.")
        ingreds.append(ing)

    eff_flat = compute_effectiveness(ingreds)
    crafted = compute_crafted_stats(recipe, decoded["tier"], ingreds,
                                    eff_flat, atk_speed=atk_speed)

    query_result = None
    if user_query is not None:
        query_result = score_query(crafted, user_query)

    print_decoded(decoded, recipe, ingreds, eff_flat, crafted, query_result)
    decoded.pop("_cursor", None)
    return {
        "decoded": decoded,
        "recipe": recipe,
        "ingredients": ingreds,
        "effectiveness": eff_flat,
        "crafted": crafted,
        "query_result": query_result,
    }


# ============================================================

dps = 497
pctW = 1000
def raw(pct, dps=dps) : return 10*pct/dps 
def is_null(val, epsilon=0.6) : return abs(val) < epsilon

nScale = 396
eScale = 80
tScale = 0
wScale = 0
fScale = 0
aScale = 28

nDmgWeaponMin = 0
nDmgWeaponMax = 0
eDmgWeaponMin = 0
eDmgWeaponMax = 0
tDmgWeaponMin = 0
tDmgWeaponMax = 0
wDmgWeaponMin = 0
wDmgWeaponMax = 0
fDmgWeaponMin = 0
fDmgWeaponMax = 0
aDmgWeaponMin = 209
aDmgWeaponMax = 275

nWeaponAvg = nDmgWeaponMin + ((nDmgWeaponMax - nDmgWeaponMin) / 2)
eWeaponAvg = eDmgWeaponMin + ((eDmgWeaponMax - eDmgWeaponMin) / 2)
tWeaponAvg = tDmgWeaponMin + ((tDmgWeaponMax - tDmgWeaponMin) / 2)
wWeaponAvg = wDmgWeaponMin + ((wDmgWeaponMax - wDmgWeaponMin) / 2)
fWeaponAvg = fDmgWeaponMin + ((fDmgWeaponMax - fDmgWeaponMin) / 2)
aWeaponAvg = aDmgWeaponMin + ((aDmgWeaponMax - aDmgWeaponMin) / 2)

nDmg = nWeaponAvg * nScale
eDmg = eWeaponAvg * nScale + nWeaponAvg * eScale + eWeaponAvg * eScale + tWeaponAvg * eScale + wWeaponAvg * eScale + fWeaponAvg * eScale + aWeaponAvg * eScale
tDmg = tWeaponAvg * nScale + nWeaponAvg * tScale + eWeaponAvg * tScale + tWeaponAvg * tScale + wWeaponAvg * tScale + fWeaponAvg * tScale + aWeaponAvg * tScale
wDmg = wWeaponAvg * nScale + nWeaponAvg * wScale + eWeaponAvg * wScale + tWeaponAvg * wScale + wWeaponAvg * wScale + fWeaponAvg * wScale + aWeaponAvg * wScale
fDmg = fWeaponAvg * nScale + nWeaponAvg * fScale + eWeaponAvg * fScale + tWeaponAvg * fScale + wWeaponAvg * fScale + fWeaponAvg * fScale + aWeaponAvg * fScale
aDmg = aWeaponAvg * nScale + nWeaponAvg * aScale + eWeaponAvg * aScale + tWeaponAvg * aScale + wWeaponAvg * aScale + fWeaponAvg * aScale + aWeaponAvg * aScale

totalDmg = nDmg + eDmg + tDmg + wDmg + fDmg + aDmg
if (totalDmg == 0):
    totalScale = 1
nWeight = pctW * nDmg / totalDmg
eWeight = pctW * eDmg / totalDmg
tWeight = pctW * tDmg / totalDmg
wWeight = pctW * wDmg / totalDmg
fWeight = pctW * fDmg / totalDmg
aWeight = pctW * aDmg / totalDmg

if __name__ == "__main__":

    URL = "https://wynnbuilder-beta.github.io/crafter/#4utKUGQut4muc4e81"

    # Same shape as main.py's user_query — set USER_QUERY = None to skip scoring.
    USER_QUERY = {
        # ===== General Damage =====
        "damPct":   {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "damRaw":   {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},
        "sdPct":    {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "sdRaw":    {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},

        # ===== Neutral =====
        "nDamPct":  {"ingredient_filter": False if is_null(nWeight) else True, "weight": nWeight},
        "nDamRaw":  {"ingredient_filter": False if is_null(nWeight) else True, "weight": raw(nWeight)},

        # ===== Earth =====
        "eDamPct":  {"ingredient_filter": False if is_null(eWeight) else True, "weight": eWeight},
        "eDamRaw":  {"ingredient_filter": False if is_null(eWeight) else True, "weight": raw(eWeight)},
        "eSdPct":   {"ingredient_filter": False if is_null(eWeight) else True, "weight": eWeight},
        "eSdRaw":   {"ingredient_filter": False if is_null(eWeight) else True, "weight": raw(eWeight)},

        # ===== Thunder =====
        "tDamPct":  {"ingredient_filter": False if is_null(tWeight) else True, "weight": tWeight},
        "tDamRaw":  {"ingredient_filter": False if is_null(tWeight) else True, "weight": raw(tWeight)},
        "tSdPct":   {"ingredient_filter": False if is_null(tWeight) else True, "weight": tWeight},
        "tSdRaw":   {"ingredient_filter": False if is_null(tWeight) else True, "weight": raw(tWeight)},

        # ===== Water =====
        "wDamPct":  {"ingredient_filter": False if is_null(wWeight) else True, "weight": wWeight},
        "wDamRaw":  {"ingredient_filter": False if is_null(wWeight) else True, "weight": raw(wWeight)},
        "wSdPct":   {"ingredient_filter": False if is_null(wWeight) else True, "weight": wWeight},
        "wSdRaw":   {"ingredient_filter": False if is_null(wWeight) else True, "weight": raw(wWeight)},

        # ===== Fire =====
        "fDamPct":  {"ingredient_filter": False if is_null(fWeight) else True, "weight": fWeight},
        "fDamRaw":  {"ingredient_filter": False if is_null(fWeight) else True, "weight": raw(fWeight)},
        "fSdPct":   {"ingredient_filter": False if is_null(fWeight) else True, "weight": fWeight},
        "fSdRaw":   {"ingredient_filter": False if is_null(fWeight) else True, "weight": raw(fWeight)},

        # ===== Air =====
        "aDamPct":  {"ingredient_filter": False if is_null(aWeight) else True, "weight": aWeight},
        "aDamRaw":  {"ingredient_filter": False if is_null(aWeight) else True, "weight": raw(aWeight)},
        "aSdPct":   {"ingredient_filter": False if is_null(aWeight) else True, "weight": aWeight},
        "aSdRaw":   {"ingredient_filter": False if is_null(aWeight) else True, "weight": raw(aWeight)},

        # ===== Rainbow ===== // BE CAREFUL IF NEUTRAL WEAPON (RARE)
        "rDamPct":  {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "rDamRaw":  {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},
        "rSdPct":   {"ingredient_filter": False if is_null(pctW) else True, "weight": pctW},
        "rSdRaw":   {"ingredient_filter": False if is_null(pctW) else True, "weight": raw(pctW)},


        # ===== Skill points =====
        "str": {"min": 0, "weight":1.5*pctW, "ingredient_filter":False},
        "dex": {"min": 0, "weight":1.5*pctW, "ingredient_filter":False},
        "int": {"min": 0, "weight":pctW, "ingredient_filter":False},
        "def": {"min": 0, "weight":pctW, "ingredient_filter":False},
        "agi": {"min": 0, "weight":0, "ingredient_filter":False},

        # ===== Sustain =====
        "mr": {"min": 0, "weight":1*pctW},
        #"ms": {"min": 0, "weight":0},
        
        #"hprRaw": {"min": -200, "ingredient_filter":False},
        #"hpBonus": {"min": 2000, "ingredient_filter":False},

        # ===== Requirements =====
        "strReq": {"max": 45, "ingredient_filter":False},
        "dexReq": {"max": 45, "ingredient_filter":False},
        "intReq": {"max": 45, "ingredient_filter":False},
        "defReq": {"max": 15, "ingredient_filter":False},
        "agiReq": {"max": 125, "ingredient_filter":True},

        # ===== Usability =====
        #"spd": {"min": 0, "weight": 0.5},

        "durability": {"min": 75, "weight": 1},
        # "duration": {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
        # "charges":  {"min": 0, "max": 0, "ingredient_filter": True, "weight": 0},
    }

    decode_and_report(URL, USER_QUERY)
