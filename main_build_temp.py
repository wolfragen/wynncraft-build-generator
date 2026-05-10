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
    REQ_STATS, REQ_STATS_IDX,
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
# v23 (= 2.1.6.0) uses 5 elements × 6 tiers = 30 powders → 5 bits per id.
# Wynnbuilder-beta versions (v >= 24) ship with 7 tiers → 6 bits per id.
# Resolved per-URL via `_powder_consts(data_version)` below.
POWDER_REPEAT_OP_BITLEN = 1
POWDER_REPEAT_TIER_OP_BITLEN = 1
POWDER_CHANGE_OP_BITLEN = 1
POWDER_WRAPPER_BITLEN = 2

# Threshold at which the beta switches to 7 tiers / 6-bit ids.
_BETA_POWDER_VERSION = 24


def _powder_consts(data_version):
    """Return (id_bitlen, num_tiers) for the given build URL data version."""
    if data_version is not None and data_version >= _BETA_POWDER_VERSION:
        return 6, 7
    return 5, 6


# Default tier count used when no per-slot value is available (e.g., the
# user injected a slot dict by hand). Equals the length of `_POWDER_STATS`
# per element; ids beyond this are mapped via `_powder_stats_lookup`.
_KNOWN_POWDER_TIERS = 6

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


POWDER_ELEMENTS = ("e", "t", "w", "f", "a")  # mirrors skp_elements in build_utils.js
NUM_POWDER_ELEMENTS = len(POWDER_ELEMENTS)


def _decode_powders(cursor, id_bitlen, num_tiers):
    """
    Walks the wynnbuilder powder encoding (mirrors `decodePowders` in
    `build_encode_decode.js`) and returns the decoded list of powder ids.

    `id_bitlen` and `num_tiers` are version-dependent (see `_powder_consts`).
    Returned ids encode both element (id // num_tiers) and tier index
    (id % num_tiers).
    """
    powders = [cursor.read(id_bitlen)]  # first powder
    prev = powders[0]
    while True:
        op = cursor.read(POWDER_REPEAT_OP_BITLEN)
        if op == 0:  # REPEAT
            powders.append(prev)
            continue
        # NO_REPEAT
        op2 = cursor.read(POWDER_REPEAT_TIER_OP_BITLEN)
        if op2 == 0:  # REPEAT_TIER — element shifts forward by `wrap+1`
            wrap = cursor.read(POWDER_WRAPPER_BITLEN)
            prev_elem = prev // num_tiers
            prev_tier = prev % num_tiers
            new_elem = (prev_elem + wrap + 1) % NUM_POWDER_ELEMENTS
            powder = new_elem * num_tiers + prev_tier
            powders.append(powder)
            prev = powder
            continue
        # CHANGE_POWDER
        op3 = cursor.read(POWDER_CHANGE_OP_BITLEN)
        if op3 == 0:  # NEW_POWDER
            powder = cursor.read(id_bitlen)
            powders.append(powder)
            prev = powder
            continue
        # NEW_ITEM
        return powders


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
                pwd_id_bits, pwd_tiers = _powder_consts(data_version)
                powders = _decode_powders(cur, pwd_id_bits, pwd_tiers)
                # User asked to take weapon powders into account during load.
                # Other slots' powders (helmet/chestplate/leggings/boots) are
                # decoded but not stored — adding their elemental defenses is
                # out of scope for this generator.
                if i == 8 and slots[-1] is not None:
                    slots[-1]["powders"] = powders
                    # Stash the tier system used so later code can interpret
                    # the ids consistently (powder id 12 means W1 in 6-tier,
                    # T6 in 7-tier — identical bytes, different semantics).
                    slots[-1]["powder_tiers"] = pwd_tiers

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


def _emit_powders(writer, powders, id_bitlen, num_tiers):
    """
    Encode a list of powder ids using the same compact run/element scheme as
    `encodePowders` in build_encode_decode.js. Inverse of `_decode_powders`.
    `id_bitlen` and `num_tiers` must match the values the URL was decoded with.
    """
    writer.emit(1, 1)  # HAS_POWDERS
    writer.emit(powders[0], id_bitlen)
    prev = powders[0]
    for p in powders[1:]:
        if p == prev:
            writer.emit(0, POWDER_REPEAT_OP_BITLEN)  # REPEAT
            continue
        writer.emit(1, POWDER_REPEAT_OP_BITLEN)  # NO_REPEAT
        if p % num_tiers == prev % num_tiers:
            writer.emit(0, POWDER_REPEAT_TIER_OP_BITLEN)  # REPEAT_TIER
            prev_elem = prev // num_tiers
            new_elem = p // num_tiers
            wrap = (new_elem - prev_elem) % NUM_POWDER_ELEMENTS - 1
            writer.emit(wrap, POWDER_WRAPPER_BITLEN)
        else:
            writer.emit(1, POWDER_REPEAT_TIER_OP_BITLEN)  # CHANGE_POWDER
            writer.emit(0, POWDER_CHANGE_OP_BITLEN)       # NEW_POWDER
            writer.emit(p, id_bitlen)
        prev = p
    # Terminate: NO_REPEAT + CHANGE_POWDER + NEW_ITEM
    writer.emit(1, POWDER_REPEAT_OP_BITLEN)
    writer.emit(1, POWDER_REPEAT_TIER_OP_BITLEN)
    writer.emit(1, POWDER_CHANGE_OP_BITLEN)


def encode_build_url(data_version, slots, recipes_by_id,
                     base="https://wynnbuilder-beta.github.io/builder/"):
    """
    Re-encode the 9 equipment slots into a complete wynnbuilder builder URL.

    Tomes/SP/level/aspects/atree are written as their "default" sentinels
    (NO_TOMES, AUTOMATIC SP, MAX level, NO_ASPECTS, empty atree). For
    powderable slots, weapon powders (slot 8) are preserved if `decode_build_url`
    captured them; other slots are written as NO_POWDERS since we don't track
    them.
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
            powders = (slot or {}).get("powders") if slot else None
            # Only weapon powders (slot 8) are decoded/preserved; for armor
            # slots we discarded powders during decode → emit NO_POWDERS.
            if i == 8 and powders:
                pwd_id_bits, pwd_tiers = _powder_consts(data_version)
                _emit_powders(w, powders, pwd_id_bits, pwd_tiers)
            else:
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


def _parse_dam_range(s):
    """Parse a 'min-max' wynnbuilder damage string. Returns (0, 0) for empty."""
    if not s:
        return 0, 0
    try:
        lo, hi = s.split("-", 1)
        return int(lo), int(hi)
    except (ValueError, AttributeError):
        return 0, 0


_WEAPON_DAM_FIELDS = (
    ("nDam", "nDamRaw"),
    ("eDam", "eDamRaw"),
    ("tDam", "tDamRaw"),
    ("wDam", "wDamRaw"),
    ("fDam", "fDamRaw"),
    ("aDam", "aDamRaw"),
)


def _accumulate_normal_item(item, base_min, base_max, powders=None,
                            powder_tiers=_KNOWN_POWDER_TIERS):
    """Add a normal item's stat contributions into base_min / base_max
    (length-STAT_COUNT int arrays). For weapons, the weapon's intrinsic
    damage ranges (nDam / eDam / ...) are folded into the corresponding
    *DamRaw stats so the spell formula sees them. Optional `powders` then
    shifts part of the neutral pool into elemental damages — `powder_tiers`
    is the tier count used by the URL's data version."""
    if item is None:
        return
    for stat_name, idx in STAT_INDEX.items():
        if stat_name in ("durability", "duration", "charges"):
            continue  # not contributed by equipped items
        # Skill-point requirements aggregate as MAX across pieces, not SUM
        # (skillpoints.js:can_equip checks each item's req individually). The
        # search engine adds candidate stats to base, so summing equipped reqs
        # here would inflate the build's effective req. Drop them entirely:
        # the user's req max constraint then applies to the new craft alone,
        # which is the right semantic when all equipped pieces also satisfy it.
        if idx in REQ_STATS_IDX:
            continue
        lo, hi = _normal_item_stat_range(item, stat_name)
        base_min[idx] += lo
        base_max[idx] += hi

    # Weapon intrinsic damage → fold into *DamRaw so it participates in spell
    # formulas. Mirrors what `_accumulate_crafted_item` does with
    # healthOrDamage * matmult * atk-speed-ratio for crafted weapons.
    if item.get("category") == "weapon":
        n_lo, n_hi = 0, 0
        for dam_field, stat_name in _WEAPON_DAM_FIELDS:
            lo, hi = _parse_dam_range(item.get(dam_field))
            if lo == 0 and hi == 0:
                continue
            if dam_field == "nDam":
                # Defer adding neutral damage until after powder conversion
                # consumes part of it.
                n_lo, n_hi = lo, hi
            else:
                idx = STAT_INDEX[stat_name]
                base_min[idx] += lo
                base_max[idx] += hi

        # Apply powders to the weapon's neutral pool (conversion + raw bonus).
        n_lo, n_hi = _apply_weapon_powders(
            powders or [], n_lo, n_hi, base_min, base_max,
            num_tiers=powder_tiers,
        )
        if n_lo or n_hi:
            idx = STAT_INDEX["nDamRaw"]
            base_min[idx] += n_lo
            base_max[idx] += n_hi


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


# Mirrors `powderStats` in js/powders.js (E,T,W,F,A × tiers 1..6).
# Tuple = (min_dmg, max_dmg, convert_pct, def_plus, def_minus). We only use
# (min_dmg, max_dmg, convert_pct) for weapon application.
_POWDER_STATS = [
    # E (idx 0..5)
    (3,6,17,2,1), (5,8,21,4,2), (6,10,25,8,3), (7,10,31,14,5), (9,11,38,22,9), (11,13,46,30,13),
    # T (idx 6..11)
    (1,8,9,3,1), (1,12,11,5,1), (2,15,13,9,2), (3,15,17,14,4), (4,17,22,20,7), (5,20,28,28,10),
    # W (idx 12..17)
    (3,4,13,3,1), (4,6,15,6,1), (5,8,17,11,2), (6,8,21,18,4), (7,10,26,28,7), (9,11,32,40,10),
    # F (idx 18..23)
    (2,5,14,3,1), (4,8,16,5,2), (5,9,19,9,3), (6,9,24,16,5), (8,10,30,25,9), (10,12,37,36,13),
    # A (idx 24..29)
    (2,6,11,3,1), (3,10,14,6,2), (4,11,17,10,3), (5,11,22,16,5), (7,12,28,24,9), (8,14,35,34,13),
]
_ELEMENT_DAMRAW_STAT = ("eDamRaw", "tDamRaw", "wDamRaw", "fDamRaw", "aDamRaw")


def _powder_stats_lookup(pid, num_tiers):
    """
    Translate a powder id from its native `num_tiers` numbering to
    (element_idx, stat_tuple) using my 6-tier `_POWDER_STATS` table.

    For tiers we don't have stats for (e.g. T7 in beta, tier_idx >= 6),
    return None so the caller can skip them with a warning.
    """
    elem = pid // num_tiers
    tier_idx = pid % num_tiers
    if tier_idx >= _KNOWN_POWDER_TIERS:
        return elem, None
    canonical = elem * _KNOWN_POWDER_TIERS + tier_idx
    if canonical >= len(_POWDER_STATS):
        return elem, None
    return elem, _POWDER_STATS[canonical]


def _apply_weapon_powders(powders, n_lo, n_hi, base_min, base_max,
                          num_tiers=_KNOWN_POWDER_TIERS, do_conversion=True):
    """
    Apply weapon powders to a weapon's intrinsic neutral damage range,
    mirroring `calc_weapon_powder` in js/powders.js.

    Logic:
      - Group powders by element; for each element, sum their raw min/max
        damage and conversion%.
      - If `do_conversion`, convert min(remaining_neutral, conv*remaining)
        from the neutral pool into that element's damage. (Disable for
        normal weapons since we don't carry their intrinsic nDam in nDamRaw —
        a fake conversion would underflow.)
      - Add the element's summed raw min/max on top.

    `num_tiers` is the powder-tier count used by the URL's data version
    (6 for v23 / mainline, 7 for wynnbuilder-beta).

    Mutates `base_min` / `base_max` for elemental DamRaw stats; returns the
    leftover (n_lo, n_hi) that should still be added to nDamRaw.
    """
    if not powders:
        return n_lo, n_hi

    # Aggregate per element while preserving first-seen order.
    apply_order = []
    apply_map = {}  # element_idx → {"conv": float, "min": int, "max": int}
    for pid in powders:
        elem, stats = _powder_stats_lookup(pid, num_tiers)
        if stats is None:
            print(f"  [warn] no powder stats for id={pid} ({num_tiers}-tier system); skipping")
            continue
        pmin, pmax, pconv, _dp, _dm = stats
        info = apply_map.get(elem)
        if info is None:
            info = {"conv": 0.0, "min": 0, "max": 0}
            apply_map[elem] = info
            apply_order.append(elem)
        info["conv"] += pconv / 100.0
        info["min"] += pmin
        info["max"] += pmax

    rem_lo = n_lo
    rem_hi = n_hi
    for elem in apply_order:
        info = apply_map[elem]
        if do_conversion:
            conv = info["conv"]
            # min(remaining, conv*remaining) — when conv ≤ 1 → conv*remaining;
            # when conv > 1 → remaining (full conversion). Matches the JS branch.
            diff_lo = min(rem_lo, int(conv * rem_lo))
            diff_hi = min(rem_hi, int(conv * rem_hi))
            rem_lo -= diff_lo
            rem_hi -= diff_hi
        else:
            diff_lo = diff_hi = 0
        stat_idx = STAT_INDEX[_ELEMENT_DAMRAW_STAT[elem]]
        base_min[stat_idx] += diff_lo + info["min"]
        base_max[stat_idx] += diff_hi + info["max"]

    return rem_lo, rem_hi


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

    # Add the rolled-id and itemIDs sums (already aggregated). Skip req
    # stats: they aggregate as MAX across pieces, not SUM. Inside a single
    # craft the per-ingredient sum above gives the piece's strReq/dexReq/
    # etc., but folding that into a multi-piece base_min/max would inflate
    # the build's effective req (see _accumulate_normal_item for details).
    for idx in range(STAT_COUNT):
        if idx in REQ_STATS_IDX:
            continue
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
            # Apply weapon powders (if any) — moves part of neutral damage to
            # elemental and adds powder raw min/max. Returns leftover neutral.
            powders = crafted.get("powders") or []
            num_tiers = crafted.get("powder_tiers", _KNOWN_POWDER_TIERS)
            n_lo, n_hi = _apply_weapon_powders(
                powders, n_lo, n_hi, base_min, base_max, num_tiers=num_tiers,
            )
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
            _accumulate_normal_item(
                item, base_min, base_max,
                powders=slot.get("powders"),
                powder_tiers=slot.get("powder_tiers", _KNOWN_POWDER_TIERS),
            )
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

def _piece_req_value(slot, items_by_id, ings_by_id):
    """
    Return {req_name: int} for one slot's per-stat skill requirement. For
    crafted pieces this sums per-ingredient itemIDs reqs with effectiveness
    (mirroring `_accumulate_crafted_item`'s inner loop). Consumables and
    empty slots return all zeros — they don't consume player skillpoints.
    """
    out = {r: 0 for r in REQ_STATS}
    if slot is None:
        return out
    kind = slot.get("kind")
    if kind == "normal":
        item = items_by_id.get(slot["id"])
        if item is None:
            return out
        for r in REQ_STATS:
            out[r] = int(item.get(r, 0) or 0)
        return out
    if kind != "crafted":
        return out
    recipe = slot.get("recipe")
    if recipe is None:
        return out
    rtype = recipe["type"].lower()
    if rtype in ("potion", "scroll", "food"):
        return out  # consumables don't have skill reqs
    ing_objs = [ings_by_id.get(iid) for iid in slot["ing_ids"]]
    pos_mods = []
    for ing in ing_objs:
        if ing is None:
            pos_mods.append([0] * 6)
        else:
            raw_pm = ing.get("posMods", {})
            pos_mods.append([raw_pm.get(k, 0) for k in POSMOD_ORDER])
    eff_flat = _compute_effectiveness(pos_mods)
    for n, ing in enumerate(ing_objs):
        if ing is None:
            continue
        eff_mult = eff_flat[n] / 100.0
        is_powder = ing.get("isPowder", False)
        for key, value in ing.get("itemIDs", {}).items():
            if key not in REQ_STATS:
                continue
            v = value if is_powder else round(value * eff_mult)
            out[key] += int(v)
    return out


def _max_equipped_reqs(slots, items_by_id, ings_by_id):
    """
    MAX of each skill req across all non-empty slots. The player allocates
    max(reqs) of skillpoints to satisfy the build (skillpoints.js apply_to_fit
    + can_equip), and that allocation contributes to total str/dex/int/def/agi
    — so it has to be folded into the SP cap calc (see Query.sp_score_*).
    """
    out = {r: 0 for r in REQ_STATS}
    for slot in slots:
        piece = _piece_req_value(slot, items_by_id, ings_by_id)
        for r in REQ_STATS:
            v = piece[r]
            if v > out[r]:
                out[r] = v
    return out


def _augment_query_with_equipped_reqs(user_query, equipped_max_reqs):
    """
    Return a shallow copy of `user_query` with two augmentations driven by
    the per-attribute MAX of equipped pieces' skill reqs:

      1. Each xReq stat's `base` is raised to the max-equipped value (kept
         if higher). Feeds `Query.sp_score_req_base_proj` so the SP cap
         clamp on direct-weight scoring accounts for the player's existing
         allocation.

      2. `_context["xReq"]` is set to the same value. Feeds
         `BUILD_CTX_BASE_REQ_*` so the spell-composite formula folds the
         existing allocation into the player's total SP via
         `s_str = ctx[BASE_STR] + deps[strBonus] + max(deps[strReq], ctx[BASE_REQ_STR])`,
         which matches `skillpoints.js apply_to_fit` semantics — the
         player allocates max(reqs across items) and that allocation
         contributes to total SP (and the 150 cap).
    """
    augmented = dict(user_query)
    ctx_existing = augmented.get("_context") or {}
    ctx_new = dict(ctx_existing)
    for req_name, max_req in equipped_max_reqs.items():
        if max_req <= 0:
            continue
        cfg = dict(augmented.get(req_name, {}))
        existing_base = int(cfg.get("base", 0) or 0)
        cfg["base"] = max(existing_base, int(max_req))
        augmented[req_name] = cfg
        existing_ctx_req = int(ctx_new.get(req_name, 0) or 0)
        ctx_new[req_name] = max(existing_ctx_req, int(max_req))
    if ctx_new != ctx_existing:
        augmented["_context"] = ctx_new
    return augmented


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


# ============================================================
# WB-faithful spell damage evaluator
# ============================================================
# Reproduces wynnbuilder's `calculateSpellDamage` (js/damage_calc.js) — the
# search-engine kernel collapses weapon damage and per-element DamRaw IDs
# into a single dep slot for speed (and B&B monotonicity), which is correct
# for *finding* the optimal craft but reports inflated damage for elements
# the spell doesn't actually hit. This evaluator keeps weapon damage and
# gear raws separate so the reported number matches wynnbuilder's display.

# Element order used by WB: N, E, T, W, F, A.
_WB_ELEM_NAMES = ("n", "e", "t", "w", "f", "a")


def _crafted_weapon_damage(slot, ings_by_id):
    """
    Compute a crafted weapon's post-powder per-element damage range.
    Returns: (damages, present)
      - damages: [(lo_low, lo_high), ...] for each of N,E,T,W,F,A — these
        are the [floor(base*0.9), floor(base*1.1)] ranges that WB shows.
      - present: bool[6] — wynnbuilder treats elements with non-zero max
        as "present" (qualifies them for raw-ID bonuses).

    Mirrors craft.js initCraftStats + apply_weapon_powders for crafted weapons.
    """
    recipe = slot.get("recipe")
    if recipe is None or "healthOrDamage" not in recipe:
        return [(0, 0)] * 6, [False] * 6

    # 1) Base unrolled neutral damage.
    mat_tiers = slot["mat_tiers"]
    amounts = [m["amount"] for m in recipe["materials"]]
    matmult = (TIER_MULT[mat_tiers[0]] * amounts[0]
               + TIER_MULT[mat_tiers[1]] * amounts[1]) / (amounts[0] + amounts[1])
    hod = recipe["healthOrDamage"]
    n_base_low  = int(hod["minimum"] * matmult)
    n_base_high = int(hod["maximum"] * matmult)

    # 2) Atk-speed ratio (matches craft.js lines 327-336).
    atk = (slot.get("atk_speed") or "NORMAL").upper()
    ratio = 2.05
    if atk == "SLOW":   ratio /= 1.5
    elif atk == "NORMAL": ratio = 1.0
    elif atk == "FAST": ratio /= 2.5
    n_base_low  = int(n_base_low  * ratio)
    n_base_high = int(n_base_high * ratio)

    # 3) Apply powders to BOTH the low_call and high_call neutral pools — JS
    #    runs `calc_weapon_powder` twice, once on each base. Each pool starts
    #    at [floor(b*0.9), floor(b*1.1)] then powder conversion mutates it.
    powders = slot.get("powders") or []
    num_tiers = slot.get("powder_tiers", _KNOWN_POWDER_TIERS)
    low_call  = _wb_calc_weapon_powder(n_base_low,  powders, num_tiers)
    high_call = _wb_calc_weapon_powder(n_base_high, powders, num_tiers)

    # 4) Final 6-element damage ranges: low_call → low pair, high_call → high pair.
    #    But WB uses (low_call → low_low ranges) for nDamLow and high_call for nDam.
    #    For our purposes, average together: damages[i] = (low_call[i][0], high_call[i][1]).
    damages = []
    present = []
    for i in range(6):
        lo = low_call[i][0]
        hi = high_call[i][1]
        damages.append((lo, hi))
        present.append(hi > 0 or (i == 0 and lo > 0))
    return damages, present


def _wb_calc_weapon_powder(n_base, powders, num_tiers):
    """
    JS calc_weapon_powder, called per base value (low or high). Returns a
    6-element list of [min, max] floats: [neutral, e, t, w, f, a].

    Starts with neutral pool = [floor(n_base*0.9), floor(n_base*1.1)] and
    elemental pools = [0, 0]. Powders convert from neutral and add raw.
    """
    damages = [
        [int(n_base * 0.9), int(n_base * 1.1)],
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    ]
    if not powders:
        return damages

    # Aggregate per element while preserving first-seen order.
    apply_order = []
    apply_map = {}
    for pid in powders:
        elem, stats = _powder_stats_lookup(pid, num_tiers)
        if stats is None:
            continue
        pmin, pmax, pconv, _dp, _dm = stats
        info = apply_map.get(elem)
        if info is None:
            info = {"conv": 0.0, "min": 0, "max": 0}
            apply_map[elem] = info
            apply_order.append(elem)
        info["conv"] += pconv / 100.0
        info["min"] += pmin
        info["max"] += pmax

    rem_lo, rem_hi = damages[0][0], damages[0][1]
    for elem in apply_order:
        info = apply_map[elem]
        diff_lo = min(rem_lo, info["conv"] * rem_lo)
        diff_hi = min(rem_hi, info["conv"] * rem_hi)
        rem_lo -= diff_lo
        rem_hi -= diff_hi
        damages[elem + 1][0] += diff_lo + info["min"]
        damages[elem + 1][1] += diff_hi + info["max"]
    damages[0] = [rem_lo, rem_hi]
    return damages


def _normal_weapon_damage(item):
    """Per-element [(lo, hi), ...] range and present[6] for a normal weapon."""
    fields = ("nDam", "eDam", "tDam", "wDam", "fDam", "aDam")
    damages, present = [], []
    for f in fields:
        lo, hi = _parse_dam_range(item.get(f))
        damages.append((lo, hi))
        present.append(hi > 0)
    # Powders: WB applies conversion at calc_weapon_powder time. We don't
    # currently parse powders for normal weapons in this code path; treat as
    # bare. If powders are provided, the caller should run _wb_calc_weapon_powder
    # on the neutral pool before calling this.
    return damages, present


def _aggregate_gear_ids_only(slots, items_by_id, ings_by_id):
    """
    Aggregate gear stats across ALL slots, EXCLUDING the weapon's intrinsic
    damage (healthOrDamage + powder conversion) — those are returned separately
    by `_crafted_weapon_damage` / `_normal_weapon_damage`. The result is a
    "pure ID rolls" stat vector usable as `stats` in WB's calculateSpellDamage.
    """
    base_min = np.zeros(STAT_COUNT, dtype=np.int64)
    base_max = np.zeros(STAT_COUNT, dtype=np.int64)
    for i, slot in enumerate(slots):
        if slot is None:
            continue
        kind = slot.get("kind")
        if kind == "normal":
            item = items_by_id.get(slot["id"])
            if item is None:
                continue
            # For normal weapons: just skip the weapon-intrinsic-damage path
            # by passing `powders=None` and not adding nDam/eDam/etc fields.
            for stat_name, idx in STAT_INDEX.items():
                if stat_name in ("durability", "duration", "charges"):
                    continue
                # Reqs aggregate MAX across pieces, not SUM — see
                # _accumulate_normal_item.
                if idx in REQ_STATS_IDX:
                    continue
                lo, hi = _normal_item_stat_range(item, stat_name)
                base_min[idx] += lo
                base_max[idx] += hi
        elif kind == "crafted":
            _accumulate_crafted_id_rolls_only(slot, ings_by_id, base_min, base_max)
    return base_min, base_max


def _accumulate_crafted_id_rolls_only(crafted, ings_by_id, base_min, base_max):
    """Like `_accumulate_crafted_item` but skips the recipe's healthOrDamage
    injection AND the weapon-powder conversion — only ingredient rolls get
    added. Used by the WB-faithful evaluator."""
    recipe = crafted.get("recipe")
    if recipe is None:
        return
    mat_tiers = crafted["mat_tiers"]
    amounts = [m["amount"] for m in recipe["materials"]]
    matmult = (TIER_MULT[mat_tiers[0]] * amounts[0]
               + TIER_MULT[mat_tiers[1]] * amounts[1]) / (amounts[0] + amounts[1])
    rtype = recipe["type"].lower()
    is_consumable = rtype in ("potion", "scroll", "food")

    ing_objs = [ings_by_id.get(iid) for iid in crafted["ing_ids"]]
    pos_mods = []
    for ing in ing_objs:
        if ing is None:
            pos_mods.append([0] * 6)
        else:
            raw_pm = ing.get("posMods", {})
            pos_mods.append([raw_pm.get(k, 0) for k in POSMOD_ORDER])
    eff_flat = _compute_effectiveness(pos_mods)

    for n, ing in enumerate(ing_objs):
        if ing is None:
            continue
        eff_mult = eff_flat[n] / 100.0
        is_powder = ing.get("isPowder", False)
        for key, value in ing.get("itemIDs", {}).items():
            if key == "dura" or is_consumable:
                continue
            idx = STAT_INDEX.get(key)
            if idx is None:
                continue
            # Reqs aggregate MAX across pieces, not SUM — see
            # _accumulate_normal_item. This path is called once per equipped
            # crafted piece, so skipping here keeps gear_min/max free of
            # cross-piece req inflation.
            if idx in REQ_STATS_IDX:
                continue
            v = value if is_powder else round(value * eff_mult)
            base_min[idx] += v
            base_max[idx] += v
        for key, range_obj in ing.get("ids", {}).items():
            stat_idx = STAT_INDEX.get(key if key != "dura" else "durability")
            if stat_idx is None:
                continue
            if isinstance(range_obj, dict):
                lo = range_obj.get("min", range_obj.get("minimum", 0))
                hi = range_obj.get("max", range_obj.get("maximum", 0))
            else:
                lo = hi = range_obj
            r0, r1 = int(lo * eff_mult), int(hi * eff_mult)
            if r0 > r1:
                r0, r1 = r1, r0
            base_min[stat_idx] += r0
            base_max[stat_idx] += r1


def eval_meteor_wb(slots, items_by_id, ings_by_id, build_ctx, spell_name="mage_meteor"):
    """
    Wynnbuilder-faithful spell damage evaluator.

    Returns dict with min/avg/max for non-crit, crit, and total (crit-weighted)
    damage. Mirrors `calculateSpellDamage` step-by-step on the build defined
    by `slots`. Use this for displaying expected damage to users — it matches
    the wynnbuilder UI numbers, unlike the search engine's
    `_eval_spell_corner_spell` which is intentionally simplified for B&B.
    """
    from data.spells import SPELLS, SPELL_INDEX, ATK_SPEED_MULT
    from data.skillpoint_lookup import SKP_PCT_BASE, SKP_DAMAGE_MULT, SKP_MAX

    spell = SPELLS[SPELL_INDEX[spell_name]]
    mults = spell["mults"]   # (m_n, m_e, m_t, m_w, m_f, m_a) percent values
    use_spell = spell["use_spell"]
    ignore_speed = spell["ignore_speed"]
    spec = "Sd" if use_spell else "Md"   # "Sd" or "Md" prefix for stat names

    # ---- 1. Weapon damage (post-powder) ----
    weapon = slots[8]
    if weapon is None:
        return None
    if weapon["kind"] == "crafted":
        wd, present = _crafted_weapon_damage(weapon, ings_by_id)
        atk_speed = (weapon.get("atk_speed") or "NORMAL").upper()
    elif weapon["kind"] == "normal":
        item = items_by_id.get(weapon["id"])
        if item is None:
            return None
        wd, present = _normal_weapon_damage(item)
        atk_speed = item.get("atkSpd", "NORMAL")
    else:
        return None

    # damages[i] = [min, max] (mutable). Initially = wd[i].
    damages = [[float(lo), float(hi)] for (lo, hi) in wd]

    # ---- 2. Conversions ----
    # 2.1 neutral conversion scales every weapon element. weapon_min/max sum.
    weapon_min = sum(d[0] for d in damages)
    weapon_max = sum(d[1] for d in damages)
    m_n = mults[0] / 100.0
    if m_n == 0:
        present = [False] * 6
    for i in range(6):
        damages[i][0] *= m_n
        damages[i][1] *= m_n

    # 2.2 elemental conversions (m_e..m_a). Each routes weapon-total to that
    #     element and marks it present.
    total_convert = m_n
    for i in range(1, 6):
        m_i = mults[i] / 100.0
        if m_i > 0:
            damages[i][0] += m_i * weapon_min
            damages[i][1] += m_i * weapon_max
            present[i] = True
            total_convert += m_i

    # ---- 3. Attack speed ----
    if not ignore_speed:
        atk_mult = ATK_SPEED_MULT.get(atk_speed, ATK_SPEED_MULT["NORMAL"])
        for i in range(6):
            damages[i][0] *= atk_mult
            damages[i][1] *= atk_mult

    # ---- 4. (DamAddMin/Max — not in our IDS_STATS, skip) ----

    # ---- 5. ID bonus ----
    # Aggregate gear IDs (rolls only — NOT including weapon intrinsic damage).
    g_min, g_max = _aggregate_gear_ids_only(slots, items_by_id, ings_by_id)
    g_mid = (g_min.astype(np.float64) + g_max.astype(np.float64)) / 2.0

    def gs(name):
        idx = STAT_INDEX.get(name)
        return float(g_mid[idx]) if idx is not None else 0.0

    # 5.1 % boost — per element (1 + skp_dam_mult * skpPct + static + (specPct + DamPct)/100)
    # Skill points: gear str/dex/etc. (bonus) + user-assigned (build_ctx[0..4])
    # + max(equipped req per attribute) for the player's allocated SP. The
    # allocation is the player's max-req allocation that satisfies all items'
    # xReq (skillpoints.js can_equip), and it counts toward the total SP that
    # feeds skillPointsToPercentage. Pulled from build_ctx[BASE_REQ_*]
    # (BUILD_CTX_BASE_REQ_STR..AGI = indices 13..17).
    skp_str = int(g_mid[STAT_INDEX["str"]] + build_ctx[0]) + int(build_ctx[13])
    skp_dex = int(g_mid[STAT_INDEX["dex"]] + build_ctx[1]) + int(build_ctx[14])
    skp_int = int(g_mid[STAT_INDEX["int"]] + build_ctx[2]) + int(build_ctx[15])
    skp_def = int(g_mid[STAT_INDEX["def"]] + build_ctx[3]) + int(build_ctx[16])
    skp_agi = int(g_mid[STAT_INDEX["agi"]] + build_ctx[4]) + int(build_ctx[17])
    skps = [skp_str, skp_dex, skp_int, skp_def, skp_agi]
    skps = [max(0, min(SKP_MAX, s)) for s in skps]
    # skill_boost[0] = 0 (neutral), [1..5] = element bonus from skp[i]
    skill_boost = [0.0]
    for i in range(5):
        skill_boost.append(SKP_PCT_BASE[skps[i]] * SKP_DAMAGE_MULT[i])

    static_boost = (gs(spec.lower() + "Pct") + gs("damPct")) / 100.0
    rainbow_boost_pct = (gs("r" + spec + "Pct") + gs("rDamPct")) / 100.0
    save_prop = []
    total_min = 0.0
    total_max = 0.0
    for i in range(6):
        save_prop.append(damages[i][:])
        total_min += damages[i][0]
        total_max += damages[i][1]
        elem = _WB_ELEM_NAMES[i]
        boost = 1 + skill_boost[i] + static_boost \
              + (gs(elem + spec + "Pct") + gs(elem + "DamPct")) / 100.0
        if i > 0:
            boost += rainbow_boost_pct
        damages[i][0] *= boost
        damages[i][1] *= boost

    total_elem_min = total_min - save_prop[0][0]
    total_elem_max = total_max - save_prop[0][1]

    # 5.2 Raw application — per element raw_boost (gated by present[i]),
    # plus prop_raw distributed proportionally.
    prop_raw    = gs(spec.lower() + "Raw") + gs("damRaw")
    rainbow_raw = gs("r" + spec + "Raw") + gs("rDamRaw")
    for i in range(6):
        elem = _WB_ELEM_NAMES[i]
        raw_boost = 0.0
        if present[i]:
            raw_boost = gs(elem + spec + "Raw") + gs(elem + "DamRaw")
        min_boost = raw_boost
        max_boost = raw_boost
        if total_max > 0:
            min_share = save_prop[i][0] / total_min if total_min else save_prop[i][1] / total_max
            min_boost += min_share * prop_raw
            max_boost += (save_prop[i][1] / total_max) * prop_raw
        if i != 0 and total_elem_max > 0:
            min_share_e = save_prop[i][0] / total_elem_min if total_elem_min else save_prop[i][1] / total_elem_max
            min_boost += min_share_e * rainbow_raw
            max_boost += (save_prop[i][1] / total_elem_max) * rainbow_raw
        damages[i][0] += min_boost * total_convert
        damages[i][1] += max_boost * total_convert

    # ---- 6. Strength + crit ----
    str_boost = 1 + SKP_PCT_BASE[skps[0]]
    crit_chance = SKP_PCT_BASE[skps[1]]
    crit_dam_pct = float(build_ctx[6])
    crit_mult = 1 + crit_dam_pct / 100.0

    norm_min = norm_max = crit_min = crit_max = 0.0
    per_elem_norm = []
    per_elem_crit = []
    for i in range(6):
        d0 = max(0.0, damages[i][0])
        d1 = max(0.0, damages[i][1])
        nmin = d0 * str_boost
        nmax = d1 * str_boost
        cmin = d0 * (str_boost + crit_mult)
        cmax = d1 * (str_boost + crit_mult)
        per_elem_norm.append((nmin, nmax))
        per_elem_crit.append((cmin, cmax))
        norm_min += nmin; norm_max += nmax
        crit_min += cmin; crit_max += cmax

    norm_avg = (norm_min + norm_max) / 2.0
    crit_avg = (crit_min + crit_max) / 2.0
    total_avg = (1 - crit_chance) * norm_avg + crit_chance * crit_avg

    return {
        "non_crit": (norm_min, norm_avg, norm_max),
        "crit":     (crit_min, crit_avg, crit_max),
        "total_avg": total_avg,
        "crit_chance": crit_chance,
        "per_elem_non_crit": per_elem_norm,
        "per_elem_crit": per_elem_crit,
        "elem_names": _WB_ELEM_NAMES,
        "present": present,
    }


def _build_ctx_array_from_query(user_query):
    """Build the float64 build_ctx array from the user's `_context` (or
    defaults if absent) — same logic as `query.query._build_ctx_array`,
    re-exported here so the end-of-run report doesn't need to call
    `build_query` just to grab the ctx."""
    from query.query import _build_ctx_array
    return _build_ctx_array(user_query.get("_context") if isinstance(user_query, dict) else None)


def _eval_spell_for_build(spell_name, equipped_min, equipped_max, build_ctx):
    """
    Evaluate a spell composite (e.g. "mage_meteor") on the fully-aggregated
    equipped stats. Reuses the same numba kernel the search engine uses for
    B&B corner bounds.

    Returns (lo, mid, hi) triple — damage at min rolls, midpoint, and max rolls.
    Note: spells aren't strictly monotone in skp, so lo/hi aren't a strict
    bound — they're "what you get if every gear stat lands at its min/max
    independently". Midpoint is the most representative single number.
    """
    from data.spells import SPELL_SPELL_DEPS, SPELL_MELEE_DEPS, SPELL_INDEX, SPELLS
    from core.search_engine import (
        _eval_spell_corner_spell, _eval_spell_corner_melee,
    )

    spell_id = SPELL_INDEX[spell_name]
    use_spell = SPELLS[spell_id]["use_spell"]
    deps_layout = SPELL_SPELL_DEPS if use_spell else SPELL_MELEE_DEPS
    eval_fn = _eval_spell_corner_spell if use_spell else _eval_spell_corner_melee

    deps_lo = np.zeros(len(deps_layout), dtype=np.float64)
    deps_hi = np.zeros(len(deps_layout), dtype=np.float64)
    for i, name in enumerate(deps_layout):
        sidx = STAT_INDEX[name]
        deps_lo[i] = float(equipped_min[sidx])
        deps_hi[i] = float(equipped_max[sidx])
    deps_mid = (deps_lo + deps_hi) / 2.0

    return (eval_fn(deps_lo, spell_id, build_ctx),
            eval_fn(deps_mid, spell_id, build_ctx),
            eval_fn(deps_hi, spell_id, build_ctx))


def _powder_label(pid, num_tiers=_KNOWN_POWDER_TIERS):
    """Human label like 'E3' for powder id (element + tier)."""
    elem = pid // num_tiers
    tier = pid % num_tiers + 1
    return f"{POWDER_ELEMENTS[elem].upper()}{tier}"


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
        powders = (slot or {}).get("powders") if slot else None
        if powders and name == "weapon":
            num_tiers = slot.get("powder_tiers", _KNOWN_POWDER_TIERS)
            labels = ", ".join(_powder_label(p, num_tiers) for p in powders)
            tag += f"  powders=[{labels}]"
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
    # Two views of the equipped stats:
    #   - `gear_min/max`: pure ID rolls (no weapon intrinsic damage, no
    #     powder conversion). Fed into the search engine via base_min/max
    #     so deps[0..5] inside the spell kernel = gear xDamRaw IDs only.
    #   - `equipped_min/max`: full aggregation including weapon intrinsic
    #     damage (post-powder). Used for the WB-faithful eval display only.
    gear_min, gear_max = _aggregate_gear_ids_only(slots, items_by_id, ings_by_id)
    equipped_min, equipped_max = aggregate_equipped_stats(slots, items_by_id, ings_by_id)

    # Compute weapon's per-element damage (post-powder) from slot 8. Used by
    # the search kernel via ctx.weapon_dam (see search_engine `_BCTX_WD_*`).
    weapon = slots[8]
    if weapon is None:
        wdam = (0.0,) * 6
    elif weapon["kind"] == "crafted":
        wd_pairs, _ = _crafted_weapon_damage(weapon, ings_by_id)
        # Use midpoint per element — gives consistent search bounds.
        wdam = tuple((lo + hi) / 2.0 for (lo, hi) in wd_pairs)
    elif weapon["kind"] == "normal":
        item = items_by_id.get(weapon["id"])
        if item is None:
            wdam = (0.0,) * 6
        else:
            wd_pairs, _ = _normal_weapon_damage(item)
            wdam = tuple((lo + hi) / 2.0 for (lo, hi) in wd_pairs)
    else:
        wdam = (0.0,) * 6

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

        # Fold the build's current max(req) per attribute into the query as
        # `base` for each xReq stat. Reqs aggregate via MAX (one allocation
        # satisfies all items at that attribute), and that allocation is
        # added to the player's total SP — query.py captures the base into
        # sp_score_req_base_proj so the SP cap clamp at scoring time sees
        # the right headroom (SKP_MAX - ctx - max(equipped_req, craft_req)).
        equipped_max_reqs = _max_equipped_reqs(slots, items_by_id, ings_by_id)
        per_slot_query = _augment_query_with_equipped_reqs(USER_QUERY, equipped_max_reqs)

        query = build_query(
            user_json=per_slot_query,
            search_for_inversion=True,
            item_type=item_type,
            skill=skill,
            consumable=consumable,
        )

        # Inject the loaded weapon's per-element damage into ctx so the spell
        # kernel scales it by m_n / m_e correctly (instead of leaving the
        # ctx slots at 0 and underestimating spell damage).
        from query.query import (BUILD_CTX_WD_N, BUILD_CTX_WD_E, BUILD_CTX_WD_T,
                                  BUILD_CTX_WD_W, BUILD_CTX_WD_F, BUILD_CTX_WD_A)
        for slot_idx, val in zip(
            (BUILD_CTX_WD_N, BUILD_CTX_WD_E, BUILD_CTX_WD_T,
             BUILD_CTX_WD_W, BUILD_CTX_WD_F, BUILD_CTX_WD_A), wdam):
            query.build_ctx[slot_idx] = float(val)

        recipe_raw = _find_recipe(py_recipes_raw, item_type, skill, lvl_min, lvl_max)
        recipe = build_recipe(recipe_raw, query, tier=tier)

        # Stack gear ID rolls (NOT weapon intrinsic damage) on top of recipe's
        # base. The weapon damage is exposed separately via ctx weapon_dam, so
        # mixing it into base would double-count it inside the spell kernel.
        _add_equipped_to_recipe_base(query, recipe, gear_min, gear_max)

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

        # Add this freshly-crafted piece into the running totals so subsequent
        # slots account for it. Update both views: `gear_min/max` (ID rolls
        # only, fed to the search) and `equipped_min/max` (full, used for the
        # final WB-faithful eval). Show the delta on `equipped` for clarity.
        prev_min = equipped_min.copy()
        prev_max = equipped_max.copy()
        _accumulate_crafted_item(chosen_crafted, ings_by_id, equipped_min, equipped_max)
        _accumulate_crafted_id_rolls_only(chosen_crafted, ings_by_id, gear_min, gear_max)
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

    # ---------- Spell damage estimate for the assembled build ----------
    # WB-faithful evaluator (mirrors js/damage_calc.js calculateSpellDamage).
    # Reports per-element non-crit ranges + total weighted average, matching
    # what wynnbuilder displays. Skip if no spell composite in user query.
    from data.stats import DERIVED_DEPENDENCIES
    spell_names = [n for n in USER_QUERY if n in DERIVED_DEPENDENCIES
                   and n.startswith(("mage_", "warrior_", "archer_",
                                     "assassin_", "shaman_"))]
    if spell_names:
        # Inject the assembled build's max-equipped req into _context so the
        # WB-faithful eval sees the player's allocated SP (the player must
        # allocate max(reqs) and that allocation contributes to total str/dex/
        # /etc., which the spell formula folds in via ctx[BASE_REQ_*]).
        final_max_reqs = _max_equipped_reqs(slots, items_by_id, ings_by_id)
        eval_query = _augment_query_with_equipped_reqs(USER_QUERY, final_max_reqs)
        build_ctx = _build_ctx_array_from_query(eval_query)
        ctx = USER_QUERY.get("_context")
        skp_note = ""
        if not ctx or not any(k in (ctx or {}) for k in ("str","dex","int","def","agi")):
            skp_note = "  [no _context skp set — using gear-only SP; wynnbuilder " \
                       "AUTOMATIC SP would allocate ~level points more]"
        print(f"\nSpell damage estimates (final build, current query):{skp_note}")
        for sn in spell_names:
            try:
                res = eval_meteor_wb(slots, items_by_id, ings_by_id, build_ctx, sn)
                if res is None:
                    print(f"  {sn:<22s} [no weapon equipped]")
                    continue
                nc = res["non_crit"]; cr = res["crit"]
                print(f"  {sn:<22s} non-crit avg={nc[1]:>10,.0f}  crit avg={cr[1]:>10,.0f}  "
                      f"total avg={res['total_avg']:>10,.0f}  crit%={res['crit_chance']*100:.1f}")
                for n, (lo, hi), pres in zip(res["elem_names"],
                                              res["per_elem_non_crit"],
                                              res["present"]):
                    if pres:
                        print(f"    {n}: {lo:>10,.0f} - {hi:>10,.0f}")
            except Exception as e:
                print(f"  {sn:<22s} [eval failed: {e}]")


if __name__ == "__main__":
    t0 = time()
    main()
    print(f"\nTotal elapsed: {time() - t0:.1f}s")
