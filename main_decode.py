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

# The pure craft.js-faithful functions + constants now live in craft_core
# (shared with the MILP solver). Re-imported here so this module's CLI is
# unchanged. See craft_core.py.
from craft_core import (
    TIER_MULT,
    ARMOR_TYPES,
    WEAPON_TYPES,
    ACCESSORY_TYPES,
    CONSUMABLE_TYPES,
    _load_json,
    _index_ingredients,
    _index_recipes,
    compute_effectiveness,
    _js_round,
    compute_crafted_stats,
    _eval_composite,
    score_query,
)


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
