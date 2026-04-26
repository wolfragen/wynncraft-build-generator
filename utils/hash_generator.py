"""
Crafter URL hash generator. Bit-for-bit compatible with wynnbuilder's
`encodeCraft` (`wynnbuilder.github.io-master/js/craft.js`):

  Layout (CRAFTED_ENCODING_VERSION = 2):
    1   bit  : legacy flag (= 0)
    7   bits : version (= 2)
    6*12 bits: ingredient ids (raw JSON `id`, NOT list index)
    12  bits : recipe id (raw JSON `id`)
    2*3 bits : material tiers (tier - 1, two slots)
    4   bits : attack speed (only if recipe.type ∈ weaponTypes)
    pad      : `6 - (length % 6)` zero bits — ALWAYS appended (even when
               already aligned, JS adds a full empty char so the decoder's
               matching `cursor.skip(6 - ...)` always consumes ≥1 padding).

  Base64 charset (JS `Base64.#digitsStr`, NOT standard b64):
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
    (digits, then UPPERCASE, then lowercase — different from Python's
    historical order.)

  Bit packing: `vec.append(v, n)` writes `v` into bits [length, length+n),
  little-endian within the 6-bit groups (char 0 = bits [0, 6)).

The reference behavior is enforced by `decodeCraft` reading the same
layout. See `js/craft.js:34-64` (encode), `js/craft.js:74-120` (decode).
"""

# JS digit order: '0-9 A-Z a-z + -'. Reversed from the prior Python charset.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"

# Wynnbuilder only encodes 3 atk-speed buckets for crafted weapons; BITLEN is
# 4 (room for future values, leaves 2 unused). See `CRAFTER_ENC` in craft.js.
ATTACK_SPEED_MAP = {
    "SLOW": 0,
    "NORMAL": 1,
    "FAST": 2,
}
ATTACK_SPEED_BITLEN = 4

# `recipe.get("type").toLowerCase()` against this set decides whether the
# atk-speed field is present. Source: js/build_utils.js:61.
_WEAPON_TYPES = {"wand", "spear", "bow", "dagger", "relik"}

_CRAFTED_ENCODING_VERSION = 2
_VERSION_BITLEN = 7
_ING_ID_BITLEN = 12
_RECIPE_ID_BITLEN = 12
_NUM_INGS = 6
_NUM_MATS = 2
_MAT_TIER_BITLEN = 3


class EncodingBitVector:
    __slots__ = ("value", "length")

    def __init__(self):
        self.value = 0
        self.length = 0

    def append(self, v: int, bitlen: int) -> None:
        # Match JS: low bits of `v` go at position `length`. Caller ensures
        # 0 <= v < 2**bitlen.
        self.value |= (v & ((1 << bitlen) - 1)) << self.length
        self.length += bitlen

    def to_b64(self) -> str:
        out = []
        temp = self.value
        # `length` is a multiple of 6 by construction (caller pads).
        for _ in range(self.length // 6):
            out.append(CHARSET[temp & 0x3F])
            temp >>= 6
        return "".join(out)


def generate_crafter_url(
    recipe_id: int,
    recipe_type: str,
    tier: int,
    ingredient_ids: list[int],
    atk_speed: str = "NORMAL",
    base_url: str = "https://wynnbuilder-beta.github.io/crafter/",
) -> str:
    """
    Encode a crafted item into a wynnbuilder-compatible URL.

    `recipe_id` and each entry of `ingredient_ids` must be the JSON `id`
    field as it appears in `recipes_compress.json` / `ingreds_compress.json`
    (these are the wire-format ids that wynnbuilder's `ingIDMap` /
    `recipeIDMap` are keyed on — NOT list positions).

    `recipe_type` is the recipe's `type` field (e.g. "HELMET", "WAND").
    Case-insensitive; matches `_WEAPON_TYPES` to decide whether atk-speed
    is appended.

    `tier` ∈ {1, 2, 3} (encoded as tier - 1, two identical material slots).
    `atk_speed` ∈ {"SLOW", "NORMAL", "FAST"}, ignored for non-weapons.
    """
    if len(ingredient_ids) != _NUM_INGS:
        raise ValueError(f"Crafter requires exactly {_NUM_INGS} ingredient ids.")
    if not (1 <= tier <= 3):
        raise ValueError(f"tier must be in 1..3, got {tier}")
    if atk_speed not in ATTACK_SPEED_MAP:
        raise ValueError(
            f"atk_speed must be one of {sorted(ATTACK_SPEED_MAP)}, got {atk_speed!r}"
        )

    vec = EncodingBitVector()

    # Legacy flag (0 = current encoding).
    vec.append(0, 1)
    # Encoding version.
    vec.append(_CRAFTED_ENCODING_VERSION, _VERSION_BITLEN)
    # Ingredient ids.
    for ing_id in ingredient_ids:
        vec.append(int(ing_id), _ING_ID_BITLEN)
    # Recipe id.
    vec.append(int(recipe_id), _RECIPE_ID_BITLEN)
    # Material tiers (two slots, both = tier - 1).
    mat_val = tier - 1
    for _ in range(_NUM_MATS):
        vec.append(mat_val, _MAT_TIER_BITLEN)
    # Attack speed for weapons only.
    is_weapon = recipe_type.lower() in _WEAPON_TYPES
    if is_weapon:
        vec.append(ATTACK_SPEED_MAP[atk_speed], ATTACK_SPEED_BITLEN)

    # JS pads with `6 - (length % 6)` zero bits — note this is NEVER zero, so
    # an aligned vector still gets a full extra '0' char appended. The decoder
    # skips the same amount, so encoder/decoder agree.
    vec.append(0, 6 - (vec.length % 6))

    sep = "" if base_url.endswith("#") else "#"
    return f"{base_url}{sep}{vec.to_b64()}"
