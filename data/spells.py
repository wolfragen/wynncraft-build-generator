"""
spells.py

Per-spell metadata for class spells. Each spell entry encodes the data needed
to compute its average damage from build stats: 6-element multipliers,
spell-vs-melee damage type, and whether attack-speed scaling applies.

Source of truth: wynnbuilder's `js/damage_calc.js` (calculateSpellDamage) +
`js/builder/optimize.js` (averageDamage definition).

Each spell is assigned a stable `spell_id` (the order in SPELLS). The formula
tag stored on a composite is `FORMULA_SPELL_DAMAGE_BASE + spell_id`, so the
search-engine kernel can dispatch to the right spell evaluator without an
extra SoA field.

Deps layout
-----------
All spells share one of two canonical dep tuples — `SPELL_SPELL_DEPS` for
spells using sdPct/sdRaw (use_spell=True), or `SPELL_MELEE_DEPS` for those
using mdPct/mdRaw. Reading deps in this fixed order lets a single generic
evaluator handle every spell; only the multiplier tuple differs per spell.

The canonical order is documented inline below; the kernel indexes deps by
the SLOT_* constants.
"""

import numpy as np


# ============================================================
# Canonical dep tuples
# ============================================================
# Element order is N-E-T-W-F-A throughout (matches wynnbuilder
# `damage_elements = ['n'].concat(skp_elements)` where skp_elements = e,t,w,f,a).
#
# Stats absent from IDS_STATS (because they cannot roll on crafted ingredients)
# are NOT included as deps; the evaluator hardcodes their contribution to 0:
#   - nSdPct, nSdRaw, tSdPct (spell layout)
#   - all per-elem MdPct, all rainbow Md/Sd raws (melee + rainbow raws)

SPELL_SPELL_DEPS = (
    # [0..5]   Per-element weapon damage (0 = neutral, 1..5 = e,t,w,f,a)
    "nDamRaw", "eDamRaw", "tDamRaw", "wDamRaw", "fDamRaw", "aDamRaw",
    # [6..11]  Per-element DamPct (% bonus on element-only damage)
    "nDamPct", "eDamPct", "tDamPct", "wDamPct", "fDamPct", "aDamPct",
    # [12..15] Per-element SdPct (n + t have no SdPct in registry → omitted)
    "eSdPct", "wSdPct", "fSdPct", "aSdPct",
    # [16..20] Per-element SdRaw (n has no SdRaw)
    "eSdRaw", "tSdRaw", "wSdRaw", "fSdRaw", "aSdRaw",
    # [21..23] Global spell + general bonuses
    "sdPct", "damPct", "sdRaw",
    # [24..25] Rainbow (applied to non-neutral elements only)
    "rDamPct", "rSdPct",
    # [26..30] Skill point bonuses (added to user's base from _context, capped 0..150)
    "str", "dex", "int", "def", "agi",
    # [31..35] Skill point requirements — folded into total SP via
    # `s_str = ctx[BASE_STR] + deps[strBonus] + max(deps[strReq], ctx[BASE_REQ_STR])`
    # so the player's allocation (= max req across items) counts toward the cap.
    "strReq", "dexReq", "intReq", "defReq", "agiReq",
)

SPELL_MELEE_DEPS = (
    # [0..5]   Per-element weapon damage
    "nDamRaw", "eDamRaw", "tDamRaw", "wDamRaw", "fDamRaw", "aDamRaw",
    # [6..11]  Per-element DamPct
    "nDamPct", "eDamPct", "tDamPct", "wDamPct", "fDamPct", "aDamPct",
    # [12..17] Per-element MdRaw (all 6 in registry)
    "nMdRaw", "eMdRaw", "tMdRaw", "wMdRaw", "fMdRaw", "aMdRaw",
    # [18..20] Global melee + general bonuses
    "mdPct", "damPct", "mdRaw",
    # [21]     Rainbow (no rMdPct in registry)
    "rDamPct",
    # [22..26] Skill point bonuses
    "str", "dex", "int", "def", "agi",
    # [27..31] Skill point requirements — see SPELL_SPELL_DEPS comment.
    "strReq", "dexReq", "intReq", "defReq", "agiReq",
)

# Slot constants — indexed into the per-spell dep array inside the kernel.
# Spell layout: 36 deps. Melee layout: 32 deps. Element-indexed slots use
# i ∈ 0..5 with 0=neutral, 1=earth, 2=thunder, 3=water, 4=fire, 5=air.

# Common (same offsets in both layouts)
SLOT_WEAPON_BASE   = 0   # 6 entries: weapon damage per element (i)
SLOT_ELEM_DAM_PCT  = 6   # 6 entries: <elem>DamPct per element (i)

# Spell-mode-specific
SLOT_S_ELEM_SD_PCT = 12  # 4 entries: e/w/f/a SdPct (n,t omitted)
SLOT_S_ELEM_SD_RAW = 16  # 5 entries: e/t/w/f/a SdRaw (n omitted)
SLOT_S_GPCT        = 21  # sdPct
SLOT_S_DAM_PCT     = 22  # damPct
SLOT_S_GRAW        = 23  # sdRaw
SLOT_S_R_DAM_PCT   = 24  # rDamPct
SLOT_S_R_SD_PCT    = 25  # rSdPct
SLOT_S_STR         = 26
SLOT_S_DEX         = 27
SLOT_S_INT         = 28
SLOT_S_DEF         = 29
SLOT_S_AGI         = 30
SPELL_DEP_COUNT    = 31

# Melee-mode-specific
SLOT_M_ELEM_MD_RAW = 12  # 6 entries: per-element MdRaw
SLOT_M_GPCT        = 18  # mdPct
SLOT_M_DAM_PCT     = 19  # damPct
SLOT_M_GRAW        = 20  # mdRaw
SLOT_M_R_DAM_PCT   = 21  # rDamPct
SLOT_M_STR         = 22
SLOT_M_DEX         = 23
SLOT_M_INT         = 24
SLOT_M_DEF         = 25
SLOT_M_AGI         = 26
MELEE_DEP_COUNT    = 27


# ============================================================
# Spell catalog
# ============================================================
# Order matters — spell_id = index in this list. Append new spells, never
# reorder, or you invalidate cached numba modules.
#
# Each entry:
#   name          — DERIVED_DEPENDENCIES key (e.g. "mage_meteor")
#   class_name    — for documentation / future filtering
#   display       — human-readable
#   mults         — (m_n, m_e, m_t, m_w, m_f, m_a) — % conversions per element
#                   (m_n is the neutral conversion, scales each weapon element
#                    in-place; m_i>0 routes weapon-damage-sum into element i)
#   use_spell     — True → use sdPct/sdRaw + SPELL_SPELL_DEPS layout
#                   False → use mdPct/mdRaw + SPELL_MELEE_DEPS layout
#   ignore_speed  — True → skip baseDamageMultiplier[atkSpd] (DoT/instant)

SPELLS = [
    # ---- Mage ----
    {"name": "mage_meteor",          "class_name": "mage",     "display": "Meteor",
     "mults": (330,  70,   0,   0,   0,   0), "use_spell": True,  "ignore_speed": False},
    {"name": "mage_ice_snake",       "class_name": "mage",     "display": "Ice Snake",
     "mults": (120,   0,   0,  55,   0,   0), "use_spell": True,  "ignore_speed": False},
    # ---- Warrior ----
    {"name": "warrior_bash",         "class_name": "warrior",  "display": "Bash",
     "mults": (170,  30,   0,   0,   0,   0), "use_spell": False, "ignore_speed": False},
    {"name": "warrior_uppercut",     "class_name": "warrior",  "display": "Uppercut",
     "mults": (260,  40,  40,   0,   0,   0), "use_spell": False, "ignore_speed": False},
    {"name": "warrior_war_scream",   "class_name": "warrior",  "display": "War Scream",
     "mults": ( 75,   0,   0,   0,  25,   0), "use_spell": False, "ignore_speed": False},
    # ---- Archer ----
    {"name": "archer_arrow_storm",   "class_name": "archer",   "display": "Arrow Storm",
     "mults": ( 30,   0,  10,   0,   0,   0), "use_spell": True,  "ignore_speed": False},
    {"name": "archer_arrow_bomb",    "class_name": "archer",   "display": "Arrow Bomb",
     "mults": (140,   0,   0,   0,  20,   0), "use_spell": True,  "ignore_speed": False},
    # ---- Assassin ----
    {"name": "assassin_spin_attack", "class_name": "assassin", "display": "Spin Attack",
     "mults": (120,   0,  30,   0,   0,   0), "use_spell": False, "ignore_speed": False},
    {"name": "assassin_multihit",    "class_name": "assassin", "display": "Multihit",
     "mults": ( 30,   0,   0,  10,   0,   0), "use_spell": False, "ignore_speed": False},
    {"name": "assassin_smoke_bomb",  "class_name": "assassin", "display": "Smoke Bomb",
     "mults": ( 25,   5,   0,   0,   0,   5), "use_spell": True,  "ignore_speed": True},   # DoT
    # ---- Shaman ----
    {"name": "shaman_totem",         "class_name": "shaman",   "display": "Totem (DPS)",
     "mults": (  6,   0,   0,   0,   0,   6), "use_spell": True,  "ignore_speed": True},   # DoT
    {"name": "shaman_aura",          "class_name": "shaman",   "display": "Aura",
     "mults": (150,   0,   0,  30,   0,   0), "use_spell": True,  "ignore_speed": False},
    {"name": "shaman_uproot",        "class_name": "shaman",   "display": "Uproot",
     "mults": ( 80,  30,  20,   0,   0,   0), "use_spell": False, "ignore_speed": False},
]


# ============================================================
# Derived numpy arrays (numba-friendly module-level constants)
# ============================================================

SPELL_INDEX = {s["name"]: i for i, s in enumerate(SPELLS)}
SPELL_COUNT = len(SPELLS)

# Per-element conversion mults, shape (SPELL_COUNT, 6), int32 (% values).
SPELL_MULTS = np.array([s["mults"] for s in SPELLS], dtype=np.int32)

# Pre-summed total_convert per spell (used to scale raw bonuses).
# total_convert = sum(mults)/100 — wynnbuilder line ~95.
SPELL_TOTAL_CONVERT = SPELL_MULTS.sum(axis=1).astype(np.float64) / 100.0

SPELL_USE_SPELL = np.array([s["use_spell"] for s in SPELLS], dtype=np.bool_)
SPELL_IGNORE_SPEED = np.array([s["ignore_speed"] for s in SPELLS], dtype=np.bool_)


def get_spell_deps(spell_id: int):
    """Canonical dep tuple for a spell — picks SPELL or MELEE layout."""
    return SPELL_SPELL_DEPS if SPELLS[spell_id]["use_spell"] else SPELL_MELEE_DEPS


# ============================================================
# Attack speed multipliers (from wynnbuilder build_utils.js)
# ============================================================
# The crafter doesn't know the target weapon's attack speed without context;
# user supplies via _context["atk_speed"]. These multiplier values are applied
# to spell damage when ignore_speed=False.

ATK_SPEED_MULT = {
    "SUPER_SLOW": 0.51,
    "VERY_SLOW":  0.83,
    "SLOW":       1.5,
    "NORMAL":     2.05,
    "FAST":       2.5,
    "VERY_FAST":  3.1,
    "SUPER_FAST": 4.3,
}
