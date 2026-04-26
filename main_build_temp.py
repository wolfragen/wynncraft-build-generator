"""
main_build_temp.py

Reads a wynnbuilder build URL, decodes the 9 equipment slots
(helmet, chestplate, leggings, boots, ring, ring, bracelet, necklace, weapon),
sums up the stats contributed by every non-empty slot, then iteratively
generates a crafted piece for each empty slot.

The user only writes the *target* query (constraints/weights they care about);
the script automatically:
  - decodes the URL
  - aggregates stats from all already-equipped pieces
  - injects those stats as additional `base_min/base_max` into the query so
    the crafter's solver searches for a piece that, *combined* with the rest
    of the build, satisfies the user's target
  - generates each empty slot in priority order, treating earlier-generated
    pieces as if they were already equipped for subsequent slots
"""

import json
import os
import numpy as np
from time import time

from data.ingredient_loader import load_ingredients, POSMOD_ORDER
from data.ingredient_db import IngredientDB
from data.recipe_loader import load_recipes
from data.recipe import build_recipe, TIER_MULT
from query.query import build_query
from query.ingredient_filter import filter_raw_ingredients
from utils.hash_generator import generate_crafter_url
from data.stats import (
    STAT_INDEX, STAT_COUNT, CONSU_SKILLS,
    IDX_DURABILITY, IDX_DURATION,
)
from core.search_engine import search_pipelined
from core.warmup import warm_numba


# ============================================================
# Wynnbuilder URL decoder (binary encoding, v23 = 2.1.6.0)
# ============================================================

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
CHAR_TO_INT = {c: i for i, c in enumerate(CHARSET)}

# Layout constants (mirrors data/2.1.6.0/encoding_consts.json + craft.js).
HEADER_FLAG_BITLEN = 6
HEADER_VERSION_BITLEN = 10
EQUIPMENT_NUM = 9
EQUIPMENT_KIND_BITLEN = 2
EK_NORMAL, EK_CRAFTED, EK_CUSTOM = 0, 1, 2
ITEM_ID_BITLEN = 13
CUSTOM_STR_LENGTH_BITLEN = 12

POWDERABLE_INDICES = {0, 1, 2, 3, 8}
POWDER_ID_BITLEN = 5
POWDER_REPEAT_OP_BITLEN = 1
POWDER_REPEAT_TIER_OP_BITLEN = 1
POWDER_CHANGE_OP_BITLEN = 1
POWDER_WRAPPER_BITLEN = 2

# Crafted (encodeCraft v2)
CRAFTED_VERSION_BITLEN = 7
CRAFTED_NUM_INGS = 6
CRAFTED_ING_ID_BITLEN = 12
CRAFTED_RECIPE_ID_BITLEN = 12
CRAFTED_NUM_MATS = 2
CRAFTED_MAT_TIER_BITLEN = 3
CRAFTED_ATK_SPD_BITLEN = 4
WEAPON_TYPES = {"wand", "spear", "bow", "dagger", "relik"}

SLOT_NAMES = ["helmet", "chestplate", "leggings", "boots",
              "ring1", "ring2", "bracelet", "necklace", "weapon"]


class BitCursor:
    """Reads a wynnbuilder b64 hash little-endian within 6-bit groups."""
    __slots__ = ("bits", "pos", "length")

    def __init__(self, hash_str):
        v = 0
        for i, c in enumerate(hash_str):
            v |= CHAR_TO_INT[c] << (i * 6)
        self.bits = v
        self.pos = 0
        self.length = len(hash_str) * 6

    def read(self, n):
        v = (self.bits >> self.pos) & ((1 << n) - 1)
        self.pos += n
        return v

    def skip(self, n):
        self.pos += n


def _decode_powders(cursor):
    """Walks the powder encoding, returning the powder count (we don't need
    the actual powder ids — just the right number of bits consumed)."""
    cursor.read(POWDER_ID_BITLEN)  # first powder
    while True:
        op = cursor.read(POWDER_REPEAT_OP_BITLEN)
        if op == 0:  # REPEAT
            continue
        # NO_REPEAT
        op2 = cursor.read(POWDER_REPEAT_TIER_OP_BITLEN)
        if op2 == 0:  # REPEAT_TIER
            cursor.read(POWDER_WRAPPER_BITLEN)
            continue
        # CHANGE_POWDER
        op3 = cursor.read(POWDER_CHANGE_OP_BITLEN)
        if op3 == 0:  # NEW_POWDER
            cursor.read(POWDER_ID_BITLEN)
            continue
        # NEW_ITEM
        return


def _decode_crafted(cursor):
    """
    Decode an inline crafted item from the equipment stream.
    Returns dict with ingredient ids, recipe id, mat tiers, atk speed,
    and the recipe type (used to know if atk-speed bits are present).
    """
    start = cursor.pos
    legacy = cursor.read(1)
    if legacy:
        # Legacy crafted encoding is unused on v23+; raise so we notice.
        raise NotImplementedError("Legacy crafted hash inside binary build URL.")
    cursor.read(CRAFTED_VERSION_BITLEN)
    ings = [cursor.read(CRAFTED_ING_ID_BITLEN) for _ in range(CRAFTED_NUM_INGS)]
    recipe_id = cursor.read(CRAFTED_RECIPE_ID_BITLEN)
    mat_tiers = [cursor.read(CRAFTED_MAT_TIER_BITLEN) + 1 for _ in range(CRAFTED_NUM_MATS)]

    # We don't know the recipe's type yet (lookup happens later), so we don't
    # know whether to consume atk-speed bits. Defer the decision: we read
    # both possibilities and pick the right one based on the recipe lookup.
    # Save current position; caller will resolve.
    return {
        "ing_ids": ings,
        "recipe_id": recipe_id,
        "mat_tiers": mat_tiers,
        "_pos_before_atkspd": cursor.pos,
        "_start": start,
        "atk_speed": None,
    }


def _resolve_crafted_atk_speed(cursor, crafted, recipes_by_id):
    """After we know the recipe, consume atk-speed bits if it's a weapon, then
    eat the alignment padding."""
    recipe = recipes_by_id.get(crafted["recipe_id"])
    rtype = (recipe["type"].lower() if recipe else "")
    if rtype in WEAPON_TYPES:
        atk_idx = cursor.read(CRAFTED_ATK_SPD_BITLEN)
        crafted["atk_speed"] = ["SLOW", "NORMAL", "FAST"][atk_idx] if atk_idx < 3 else "NORMAL"
    # Pad to next 6-bit boundary, as in encodeCraft.
    pad = 6 - ((cursor.pos - crafted["_start"]) % 6)
    cursor.skip(pad)
    crafted["recipe"] = recipe
    return crafted


def decode_build_url(url, recipes_by_id):
    """
    Returns: (data_version, slots).

    `slots` is a list of 9 entries. Each entry is one of:
      - None                    (empty slot)
      - {"kind": "normal", "id": <wynnbuilder item id>}
      - {"kind": "crafted", "ing_ids": [...], "recipe_id": ..., "mat_tiers": [...],
         "atk_speed": "SLOW"|"NORMAL"|"FAST"|None, "recipe": <recipe dict>}
      - {"kind": "custom"}      (we don't decode custom items)

    `data_version` is the 10-bit value from the URL header — needed so the
    re-encoded URL points at the same wynnbuilder data version.
    """
    if "#" in url:
        hash_str = url.split("#", 1)[1]
    else:
        hash_str = url
    if not hash_str:
        raise ValueError("URL has no hash payload")

    cur = BitCursor(hash_str)
    flag = cur.read(HEADER_FLAG_BITLEN)
    if flag <= 11:
        raise NotImplementedError("Legacy (non-binary) build URL not supported")
    data_version = cur.read(HEADER_VERSION_BITLEN)

    slots = []
    for i in range(EQUIPMENT_NUM):
        kind = cur.read(EQUIPMENT_KIND_BITLEN)
        if kind == EK_NORMAL:
            iid = cur.read(ITEM_ID_BITLEN)
            slots.append(None if iid == 0 else {"kind": "normal", "id": iid - 1})
        elif kind == EK_CRAFTED:
            crafted = _decode_crafted(cur)
            _resolve_crafted_atk_speed(cur, crafted, recipes_by_id)
            crafted["kind"] = "crafted"
            slots.append(crafted)
        elif kind == EK_CUSTOM:
            length = cur.read(CUSTOM_STR_LENGTH_BITLEN)
            cur.skip(length * 6)
            slots.append({"kind": "custom"})
        else:
            raise ValueError(f"Unknown equipment kind {kind}")

        if i in POWDERABLE_INDICES:
            has_powders = cur.read(1)
            if has_powders:
                _decode_powders(cur)

    return data_version, slots


# ============================================================
# URL encoder (inverse of decode_build_url)
# ============================================================

# Trailing flag bit values (mirrors data/2.1.6.0/encoding_consts.json).
_TOMES_FLAG_NO   = 0
_SP_FLAG_AUTO    = 1
_LEVEL_FLAG_MAX  = 0
_ASPECTS_FLAG_NO = 0

_ATK_SPEED_TO_INT = {"SLOW": 0, "NORMAL": 1, "FAST": 2}


class _BitWriter:
    """LSB-first bit packer matching wynnbuilder's EncodingBitVector layout.

    Bit i of the output ends up in char floor(i/6), at offset (i % 6) within
    that char's 6-bit value. `to_b64()` walks 6-bit chunks; trailing bits
    past the last full chunk are zero-extended to the next 6-bit boundary,
    same as `BitVector.toB64`."""
    __slots__ = ("value", "length")

    def __init__(self):
        self.value = 0
        self.length = 0

    def emit(self, v, n):
        self.value |= (int(v) & ((1 << n) - 1)) << self.length
        self.length += n

    def to_b64(self):
        # Round up to next 6-bit boundary; trailing partial chunk reads zero
        # bits beyond `length`, which is exactly what JS's slice() does.
        total_chars = (self.length + 5) // 6
        out = []
        temp = self.value
        for _ in range(total_chars):
            out.append(CHARSET[temp & 0x3F])
            temp >>= 6
        return "".join(out)


def _emit_crafted(writer, crafted, recipes_by_id):
    """Write a single crafted item's bits (legacy + version + ings + recipe +
    mat tiers + atk-speed-if-weapon + 6-aligned padding) into `writer`."""
    start = writer.length
    writer.emit(0, 1)  # legacy = 0
    writer.emit(2, CRAFTED_VERSION_BITLEN)  # CRAFTED_ENCODING_VERSION
    for iid in crafted["ing_ids"]:
        writer.emit(iid, CRAFTED_ING_ID_BITLEN)
    writer.emit(crafted["recipe_id"], CRAFTED_RECIPE_ID_BITLEN)
    for t in crafted["mat_tiers"]:
        writer.emit(int(t) - 1, CRAFTED_MAT_TIER_BITLEN)

    recipe = crafted.get("recipe") or recipes_by_id.get(crafted["recipe_id"])
    rtype = recipe["type"].lower() if recipe else ""
    if rtype in WEAPON_TYPES:
        atk = (crafted.get("atk_speed") or "NORMAL").upper()
        writer.emit(_ATK_SPEED_TO_INT.get(atk, 1), CRAFTED_ATK_SPD_BITLEN)

    # Pad to next 6-bit boundary (matches encodeCraft in craft.js).
    pad = 6 - ((writer.length - start) % 6)
    writer.emit(0, pad)


def crafted_to_url(crafted, recipes_by_id,
                   base="https://wynnbuilder-beta.github.io/crafter/"):
    """Standalone crafter URL for a single crafted item — same b64 payload as
    the inline crafted block, just prefixed with the crafter base URL."""
    w = _BitWriter()
    _emit_crafted(w, crafted, recipes_by_id)
    sep = "" if base.endswith("#") else "#"
    return f"{base}{sep}{w.to_b64()}"


def encode_build_url(data_version, slots, recipes_by_id,
                     base="https://wynnbuilder-beta.github.io/builder/"):
    """
    Re-encode the 9 equipment slots into a complete wynnbuilder builder URL.

    Tomes/SP/level/aspects/atree are written as their "default" sentinels
    (NO_TOMES, AUTOMATIC SP, MAX level, NO_ASPECTS, empty atree). Powders
    on powderable slots are written as NO_POWDERS — the original URL's
    powder data is discarded by `decode_build_url` and we don't reconstruct it.
    """
    w = _BitWriter()
    w.emit(0xC, HEADER_FLAG_BITLEN)            # VECTOR_FLAG
    w.emit(data_version, HEADER_VERSION_BITLEN)

    for i, slot in enumerate(slots):
        if slot is None:
            w.emit(EK_NORMAL, EQUIPMENT_KIND_BITLEN)
            w.emit(0, ITEM_ID_BITLEN)  # id=0 → empty
        elif slot["kind"] == "normal":
            w.emit(EK_NORMAL, EQUIPMENT_KIND_BITLEN)
            w.emit(int(slot["id"]) + 1, ITEM_ID_BITLEN)
        elif slot["kind"] == "crafted":
            w.emit(EK_CRAFTED, EQUIPMENT_KIND_BITLEN)
            _emit_crafted(w, slot, recipes_by_id)
        elif slot["kind"] == "custom":
            raise NotImplementedError(
                "Custom items can't be re-encoded — their bit payload was "
                "skipped during decoding."
            )
        else:
            raise ValueError(f"Unknown slot kind: {slot.get('kind')!r}")

        if i in POWDERABLE_INDICES:
            w.emit(0, 1)  # NO_POWDERS

    # Trailing minimal envelope: tomes / sp / level / aspects / atree.
    w.emit(_TOMES_FLAG_NO,   1)
    w.emit(_SP_FLAG_AUTO,    1)
    w.emit(_LEVEL_FLAG_MAX,  1)
    w.emit(_ASPECTS_FLAG_NO, 1)
    # atree empty — no bits

    sep = "" if base.endswith("#") else "#"
    return f"{base}{sep}{w.to_b64()}"


# ============================================================
# Item / recipe / ingredient lookup
# ============================================================

def load_items_by_id(path):
    """Load items.json from the wynnbuilder mirror and key by numeric id."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {it["id"]: it for it in raw["items"] if "id" in it}


def load_recipes_by_id(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {r["id"]: r for r in raw["recipes"]}


# Stat name aliases between wynnbuilder items.json and our STAT_INDEX names.
# items.json uses the same short keys for most stats; the few exceptions are
# "hp" (max hp on armor — different from "hpBonus" id) and the elemental
# defenses. We map only the IDs that overlap with STAT_INDEX.

# These items.json field names are ROLLED — i.e. their published value is the
# nominal max base, the actual roll is in [0.3x, 1.3x] (or [1.3x, 0.7x] for
# negatives). Mirror of build_utils.js:rolledIDs intersected with our stats.
_ROLLED_FIELDS = {
    "hprPct", "mr", "sdPct", "mdPct", "ls", "ms", "xpb", "lb", "ref",
    "thorns", "expd", "spd", "atkTier", "poison", "hpBonus", "eSteal",
    "hprRaw", "sdRaw", "mdRaw",
    "fDamPct", "wDamPct", "aDamPct", "tDamPct", "eDamPct",
    "fDefPct", "wDefPct", "aDefPct", "tDefPct", "eDefPct",
    "rSdRaw", "sprint", "sprintReg", "jh", "lq", "gXp", "gSpd",
    "eMdRaw", "eSdRaw", "eDamRaw",
    "tMdRaw", "tSdRaw", "tDamRaw",
    "wMdRaw", "wSdRaw", "wDamRaw",
    "fMdRaw", "fSdRaw", "fDamRaw",
    "aMdRaw", "aSdRaw", "aDamRaw",
    "nMdRaw", "nDamPct", "nDamRaw",
    "damPct",
    "rSdPct", "rDamPct",
    "healPct", "kb", "rDefPct", "maxMana",
}


def _id_round(v):
    """Match build_utils.js idRound: rounds half-away-from-zero for non-zero
    values, never producing 0 (clamps to ±1 for tiny rolls)."""
    if v == 0:
        return 0
    sign = 1 if v > 0 else -1
    rounded = int(abs(v) + 0.5) * sign
    if rounded == 0:
        rounded = sign
    return rounded


def _normal_item_stat_range(item, stat_name):
    """
    Returns (min_roll, max_roll) for `stat_name` on a normal (non-crafted) item.
    For non-rolled IDs, value is fixed; for rolled IDs, returns the [0.3x, 1.3x]
    band (or [1.3x, 0.7x] for negatives) using build_utils.js idRound rules.
    """
    raw = item.get(stat_name, 0)
    if not raw:
        return 0, 0
    if item.get("identified") or stat_name not in _ROLLED_FIELDS:
        return raw, raw
    if raw > 0:
        return _id_round(raw * 0.3), _id_round(raw * 1.3)
    # Negative: max-roll closer to zero, min-roll further. Return (min, max).
    return _id_round(raw * 1.3), _id_round(raw * 0.7)


def _accumulate_normal_item(item, base_min, base_max):
    """Add a normal item's stat contributions into base_min / base_max
    (length-STAT_COUNT int arrays)."""
    if item is None:
        return
    for stat_name, idx in STAT_INDEX.items():
        if stat_name in ("durability", "duration", "charges"):
            continue  # not contributed by equipped items
        lo, hi = _normal_item_stat_range(item, stat_name)
        base_min[idx] += lo
        base_max[idx] += hi


# ============================================================
# Crafted-piece stat reproduction (mirrors craft.js initCraftStats)
# ============================================================

# Position indices on the 3x2 ingredient grid map to the flat ingreds[0..5] as:
#   [0]=top-left, [1]=top-right, [2]=mid-left, [3]=mid-right, [4]=bot-left, [5]=bot-right
# i = floor(n/2), j = n%2  — same as craft.js.

def _compute_effectiveness(ingreds_pos_mods):
    """
    Reproduce craft.js's eff[3][2] computation.
    Input: list of 6 length-6 int arrays (POSMOD_ORDER = left,right,above,under,touching,notTouching).
    Returns: list of 6 ints (eff_flat, %).
    """
    eff = [[100, 100], [100, 100], [100, 100]]
    for n in range(6):
        i, j = n // 2, n % 2
        pm = ingreds_pos_mods[n]
        left, right, above, under, touching, not_touching = (int(pm[k]) for k in range(6))

        # left/right: target the OTHER cell in row i
        if right and j == 0:
            eff[i][1] += right
        if left and j == 1:
            eff[i][0] += left

        # above: every cell in row index < i, COLUMN-WIDE (both j positions);
        # under: every cell in row > i, COLUMN-WIDE.
        # Note: wynnbuilder craft.js does `eff[k][j] += value` which only
        # touches the SAME column. ground-truth memory: above/under = whole
        # column, but the code reads as same-column-only — which IS the same
        # column in the per-ingredient loop. We mirror craft.js exactly:
        if above:
            for k in range(i - 1, -1, -1):
                eff[k][j] += above
        if under:
            for k in range(i + 1, 3):
                eff[k][j] += under

        if touching:
            for k in range(3):
                for l in range(2):
                    if (abs(k - i) == 1 and l == j) or (k == i and abs(l - j) == 1):
                        eff[k][l] += touching
        if not_touching:
            for k in range(3):
                for l in range(2):
                    if (abs(k - i) > 1) or (abs(k - i) == 1 and abs(l - j) == 1):
                        eff[k][l] += not_touching

    return [eff[0][0], eff[0][1], eff[1][0], eff[1][1], eff[2][0], eff[2][1]]


def _accumulate_crafted_item(crafted, ings_by_id, base_min, base_max):
    """
    Compute (min_roll, max_roll) per stat for a crafted piece and add into
    base_min/base_max. Mirrors craft.js's stat aggregation:
      - per-ingredient effectiveness from posMods
      - itemIDs (reqs, dura) at floor-eff
      - rolledIDs at floor-eff for both min & max rolls
      - durability from recipe (scaled by mat tier) + per-ingredient dura,
        then floored, clamped to >=1
      - healthOrDamage scaled by mat tier (hpBonus for armor, nDamRaw for weapon).
    """
    recipe = crafted["recipe"]
    if recipe is None:
        return  # unknown recipe, skip silently

    mat_tiers = crafted["mat_tiers"]
    amounts = [m["amount"] for m in recipe["materials"]]
    matmult = (TIER_MULT[mat_tiers[0]] * amounts[0]
               + TIER_MULT[mat_tiers[1]] * amounts[1]) / (amounts[0] + amounts[1])

    rtype = recipe["type"].lower()
    is_armor = rtype in ("helmet", "chestplate", "leggings", "boots")
    is_weapon = rtype in WEAPON_TYPES

    # Resolve ingredient objects (may be None if id unknown).
    ing_objs = [ings_by_id.get(iid) for iid in crafted["ing_ids"]]

    # Effectiveness: ingredients without posMods (None / "No Ingredient") still
    # contribute zero. Pad with zero-vectors as needed.
    pos_mods = []
    for ing in ing_objs:
        if ing is None:
            pos_mods.append([0] * 6)
        else:
            raw_pm = ing.get("posMods", {})
            pos_mods.append([raw_pm.get(k, 0) for k in POSMOD_ORDER])
    eff_flat = _compute_effectiveness(pos_mods)

    is_consumable = rtype in ("potion", "scroll", "food")

    # ----- itemIDs (reqs + dura) & rolledIDs -----
    durability_min = 0
    durability_max = 0
    if not is_consumable:
        base_dura = recipe["durability"]
        durability_min += int(base_dura["minimum"] * matmult)
        durability_max += int(base_dura["maximum"] * matmult)

    # rolled stats (sum of min/max rolls per stat across ingreds), clamped via
    # the same idRound semantics as for normal items.
    add_min = np.zeros(STAT_COUNT, dtype=np.int64)
    add_max = np.zeros(STAT_COUNT, dtype=np.int64)

    for n, ing in enumerate(ing_objs):
        if ing is None:
            continue
        eff_mult = eff_flat[n] / 100.0
        is_powder = ing.get("isPowder", False)

        # itemIDs: reqs (consumables skip) + dura
        for key, value in ing.get("itemIDs", {}).items():
            if key == "dura":
                durability_min += value
                durability_max += value
                continue
            if is_consumable:
                continue
            idx = STAT_INDEX.get(key)
            if idx is None:
                continue
            if is_powder:
                add_min[idx] += value
                add_max[idx] += value
            else:
                v = round(value * eff_mult)
                add_min[idx] += v
                add_max[idx] += v

        # ids (rolled) — applied with eff_mult floor; result sorted (min, max)
        ids = ing.get("ids", {})
        for key, range_obj in ids.items():
            stat_idx = STAT_INDEX.get(key if key != "dura" else "durability")
            if stat_idx is None:
                continue
            if isinstance(range_obj, dict):
                lo = range_obj.get("min", range_obj.get("minimum", 0))
                hi = range_obj.get("max", range_obj.get("maximum", 0))
            else:
                lo = hi = range_obj
            r0 = int(lo * eff_mult)
            r1 = int(hi * eff_mult)
            if r0 > r1:
                r0, r1 = r1, r0
            add_min[stat_idx] += r0
            add_max[stat_idx] += r1

        # consumableIDs: skip — they affect duration/charges, not equipment stats.

    # Add the rolled-id and itemIDs sums (already aggregated)
    for idx in range(STAT_COUNT):
        if add_min[idx] or add_max[idx]:
            base_min[idx] += int(add_min[idx])
            base_max[idx] += int(add_max[idx])

    # Durability/duration: clamp ≥1 like craft.js (we don't track this on the
    # equipped pieces — only the slot we're crafting cares about durability).
    # Store nothing; durability is per-slot.

    # ----- healthOrDamage -----
    if "healthOrDamage" in recipe:
        hod = recipe["healthOrDamage"]
        lo = int(hod["minimum"] * matmult)
        hi = int(hod["maximum"] * matmult)
        if is_armor:
            idx = STAT_INDEX["hpBonus"]
            base_min[idx] += lo
            base_max[idx] += hi
        elif is_weapon:
            # Apply atk-speed ratio (matches craft.js).
            atk = crafted.get("atk_speed") or "NORMAL"
            ratio = 2.05
            if atk == "SLOW":
                ratio /= 1.5
            elif atk == "NORMAL":
                ratio = 1
            elif atk == "FAST":
                ratio /= 2.5
            n_lo = int(lo * ratio)
            n_hi = int(hi * ratio)
            idx = STAT_INDEX["nDamRaw"]
            base_min[idx] += n_lo
            base_max[idx] += n_hi


# ============================================================
# Aggregator: from decoded slots → equipped-stat vectors
# ============================================================

def aggregate_equipped_stats(slots, items_by_id, ings_by_id):
    """
    Returns (equipped_min, equipped_max): two np.int64 arrays of length
    STAT_COUNT, summing the stat contributions of every NON-empty slot.

    Empty slots are returned by `decode_build_url` as None and skipped here.
    """
    base_min = np.zeros(STAT_COUNT, dtype=np.int64)
    base_max = np.zeros(STAT_COUNT, dtype=np.int64)
    for slot in slots:
        if slot is None:
            continue
        kind = slot.get("kind")
        if kind == "normal":
            item = items_by_id.get(slot["id"])
            if item is None:
                print(f"  [warn] unknown wynnbuilder item id {slot['id']}, skipping")
                continue
            _accumulate_normal_item(item, base_min, base_max)
        elif kind == "crafted":
            _accumulate_crafted_item(slot, ings_by_id, base_min, base_max)
        elif kind == "custom":
            print("  [warn] custom item present in URL — its stats are NOT aggregated")
    return base_min, base_max


# ============================================================
# Per-slot crafting spec (skill / lvl / tier) for empty slots
# ============================================================

# Default crafting parameters for each empty slot. Edit per-build if needed.
# `weapon` defaults assume a wand; change to your class/weapon type.
SLOT_CRAFT_SPEC = {
    "helmet":    {"skill": "ARMOURING",      "item_type": "HELMET",     "lvl": (117, 119), "tier": 3},
    "chestplate":{"skill": "ARMOURING",      "item_type": "CHESTPLATE", "lvl": (117, 119), "tier": 3},
    "leggings":  {"skill": "TAILORING",      "item_type": "LEGGINGS",   "lvl": (117, 119), "tier": 3},
    "boots":     {"skill": "TAILORING",      "item_type": "BOOTS",      "lvl": (117, 119), "tier": 3},
    "ring1":     {"skill": "JEWELING",       "item_type": "RING",       "lvl": (117, 119), "tier": 3},
    "ring2":     {"skill": "JEWELING",       "item_type": "RING",       "lvl": (117, 119), "tier": 3},
    "bracelet":  {"skill": "JEWELING",       "item_type": "BRACELET",   "lvl": (117, 119), "tier": 3},
    "necklace":  {"skill": "JEWELING",       "item_type": "NECKLACE",   "lvl": (117, 119), "tier": 3},
    "weapon":    {"skill": "WOODWORKING",    "item_type": "WAND",       "lvl": (117, 119), "tier": 3},
}

# Search priority: weapon first, then armor (top → bottom), then accessories.
SEARCH_PRIORITY = ["weapon", "helmet", "chestplate", "leggings", "boots",
                   "ring1", "ring2", "bracelet", "necklace"]


# ============================================================
# Query augmentation
# ============================================================

def _find_recipe(recipes_raw, item_type, skill, lvl_min, lvl_max):
    """Same as data.recipe_loader.find_recipe but reusable here."""
    from data.ingredient_loader import SKILL_INDEX
    sidx = SKILL_INDEX[skill]
    for r in recipes_raw:
        if r.item_type == item_type and r.skill_index == sidx \
                and r.lvl_min == lvl_min and r.lvl_max == lvl_max:
            return r
    raise ValueError(f"Recipe not found: {skill} {item_type} {lvl_min}-{lvl_max}")


def _add_equipped_to_recipe_base(query, recipe, equipped_min, equipped_max):
    """
    Mutate `recipe.base_min_stats_proj` / `base_max_stats_proj` in place to
    add the equipped contributions on top of whatever recipe.py injected
    (durability + healthOrDamage).

    Why post-recipe mutation rather than the query's `min_base`/`max_base`?
    `recipe.py:inject_stat` uses max/min clamping when overwriting nonzero
    base slots — so for stats the recipe already touches (e.g., `hpBonus`
    for armor), my additive equipped contributions would be lost. Adding
    after `build_recipe` keeps both the recipe injection and the equipped
    sum coexisting correctly: search's effective base = recipe + equipped.
    """
    proj = query.proj_stats_idx
    skip_idx = {STAT_INDEX["durability"], STAT_INDEX["duration"], STAT_INDEX["charges"]}
    for idx_full in range(STAT_COUNT):
        if idx_full in skip_idx:
            continue
        p = proj[idx_full]
        if p == -1:
            continue
        recipe.base_min_stats_proj[p] += int(equipped_min[idx_full])
        recipe.base_max_stats_proj[p] += int(equipped_max[idx_full])


def _print_slot_summary(slots, items_by_id):
    print("Decoded equipment:")
    for name, slot in zip(SLOT_NAMES, slots):
        if slot is None:
            tag = "EMPTY"
        elif slot["kind"] == "normal":
            it = items_by_id.get(slot["id"])
            tag = it["displayName"] if it else f"UNKNOWN id={slot['id']}"
        elif slot["kind"] == "crafted":
            r = slot.get("recipe") or {}
            tag = f"CRAFTED({r.get('type','?')} {r.get('skill','?')} lvl {r.get('lvl', {}).get('minimum','?')}-{r.get('lvl', {}).get('maximum','?')})"
        else:
            tag = f"<{slot['kind']}>"
        print(f"  {name:<10s} : {tag}")


# ============================================================
# Main
# ============================================================

# Edit this URL to your build.
BUILD_URL = "https://wynnbuilder-beta.github.io/builder/#CT00000000000000000000041PCe7mBrCa9mB9BI1G6k0"

# Edit the user query — only the constraints you actually care about.
# The script auto-injects the equipped pieces' contributions as base offsets.
USER_QUERY = {
    "mage_meteor": {"weight": 100000, "ingredient_filter": True},

    "strReq": {"max": 100, "ingredient_filter": True},
    "dexReq": {"max": 100, "ingredient_filter": True},
    "intReq": {"max": 100, "ingredient_filter": True},
    "defReq": {"max": 100, "ingredient_filter": True},
    "agiReq": {"max": 100, "ingredient_filter": True},

    "durability": {"min": 40, "weight": 1},
}

# Wynnbuilder data mirror — used ONLY for items.json (the python project has
# no item index of its own). Ingredients/recipes are read from the project's
# *_compress.json files since those are newer and aligned with the search code.
WYNN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "wynnbuilder.github.io-master", "data", "2.1.6.0",
)


def main():
    t_warm = time()
    warm_numba()
    print(f"Numba warm-up: {time() - t_warm:.1f}s")

    # ---------- Load reference data ----------
    items_path = os.path.join(WYNN_DATA_DIR, "items.json")
    items_by_id = load_items_by_id(items_path)

    # Recipes & ingredients: prefer the project's compressed files (newer).
    with open("data/recipes_compress.json", "r", encoding="utf-8") as f:
        recipes_raw_dict = json.load(f)
    recipes_by_id = {r["id"]: r for r in recipes_raw_dict["recipes"]}

    with open("data/ingreds_compress.json", "r", encoding="utf-8") as f:
        ings_raw = json.load(f)
    if isinstance(ings_raw, dict):
        ings_raw = ings_raw.get("ingredients", [])
    ings_by_id = {i["id"]: i for i in ings_raw}

    # ---------- Decode URL ----------
    data_version, slots = decode_build_url(BUILD_URL, recipes_by_id)
    _print_slot_summary(slots, items_by_id)

    # ---------- Aggregate equipped stats ----------
    equipped_min, equipped_max = aggregate_equipped_stats(slots, items_by_id, ings_by_id)

    # ---------- Project python-side ingredients/recipes (used for searches) ----------
    py_ingredients_raw = load_ingredients("data/ingreds_compress.json")
    py_recipes_raw     = load_recipes("data/recipes_compress.json")
    id_to_name = {int(ing.ing_id): ing.name for ing in py_ingredients_raw}

    # ---------- Iterate empty slots in priority order ----------
    for slot_name in SEARCH_PRIORITY:
        idx = SLOT_NAMES.index(slot_name)
        if slots[idx] is not None:
            continue  # already filled (normal/crafted/custom) — skip

        spec = SLOT_CRAFT_SPEC[slot_name]
        skill = spec["skill"]
        item_type = spec["item_type"]
        lvl_min, lvl_max = spec["lvl"]
        tier = spec["tier"]
        consumable = skill in CONSU_SKILLS

        print(f"\n{'='*60}\nGenerating {slot_name} ({skill} {item_type} lvl {lvl_min}-{lvl_max} t{tier})\n{'='*60}")

        query = build_query(
            user_json=USER_QUERY,
            search_for_inversion=True,
            item_type=item_type,
            skill=skill,
            consumable=consumable,
        )

        recipe_raw = _find_recipe(py_recipes_raw, item_type, skill, lvl_min, lvl_max)
        recipe = build_recipe(recipe_raw, query, tier=tier)

        # Stack equipped stats on top of recipe's base — see helper docstring.
        _add_equipped_to_recipe_base(query, recipe, equipped_min, equipped_max)

        filtered_raw = filter_raw_ingredients(py_ingredients_raw, query, recipe)
        db = IngredientDB(filtered_raw, query)

        print(f"Filtered ingredients: {len(db)}")

        best_solution = search_pipelined(
            skill, query, recipe, db, max_cull=query.suggested_max_cull,
        )

        if best_solution is None:
            print(f"  [!] No solution found for {slot_name}")
            continue

        names = [id_to_name[int(i)] for i in best_solution]
        print(f"  Solution: {names}")

        chosen_crafted = {
            "kind": "crafted",
            "ing_ids": [int(i) for i in best_solution],
            "recipe_id": recipe_raw.data["id"],
            "mat_tiers": [tier, tier],
            # Weapon atk-speed is taken from SLOT_CRAFT_SPEC if present,
            # otherwise NORMAL. Only matters for weapons (encoded only then).
            "atk_speed": spec.get("atk_speed", "NORMAL"),
            "recipe": recipes_by_id.get(recipe_raw.data["id"]),
        }
        url = crafted_to_url(chosen_crafted, recipes_by_id)
        print(f"  Crafter URL: {url}")

        # Add this freshly-crafted piece into the equipped-stats running totals
        # so subsequent slots account for it. Show the delta so it's clear the
        # base used for the next slot's search reflects this generation.
        prev_min = equipped_min.copy()
        prev_max = equipped_max.copy()
        _accumulate_crafted_item(chosen_crafted, ings_by_id, equipped_min, equipped_max)
        slots[idx] = chosen_crafted

        delta_lines = []
        for stat_name, sidx in STAT_INDEX.items():
            if stat_name in ("durability", "duration", "charges"):
                continue
            d_lo = int(equipped_min[sidx] - prev_min[sidx])
            d_hi = int(equipped_max[sidx] - prev_max[sidx])
            if d_lo or d_hi:
                delta_lines.append(
                    f"    {stat_name:>10s}: {d_lo:+5d}..{d_hi:+5d}  "
                    f"(running base now {int(equipped_min[sidx]):+5d}..{int(equipped_max[sidx]):+5d})"
                )
        if delta_lines:
            print(f"  Stats added to base for next slot:")
            for line in delta_lines:
                print(line)

    # ---------- Final output: per-item crafter URLs + combined builder URL ----------
    print("\n" + "=" * 60)
    print("FINAL BUILD")
    print("=" * 60)
    print("\nPer-slot crafter URLs (crafted items only):")
    for name, slot in zip(SLOT_NAMES, slots):
        if slot is None:
            print(f"  {name:<10s} : (empty)")
        elif slot["kind"] == "crafted":
            print(f"  {name:<10s} : {crafted_to_url(slot, recipes_by_id)}")
        elif slot["kind"] == "normal":
            it = items_by_id.get(slot["id"])
            label = it["displayName"] if it else f"item id {slot['id']}"
            print(f"  {name:<10s} : {label} (not crafted)")
        else:
            print(f"  {name:<10s} : <{slot['kind']}>")

    final_url = encode_build_url(data_version, slots, recipes_by_id)
    print(f"\nFinal wynnbuilder URL:\n  {final_url}")


if __name__ == "__main__":
    t0 = time()
    main()
    print(f"\nTotal elapsed: {time() - t0:.1f}s")
