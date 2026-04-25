"""
spells.py

Per-spell metadata for class spells. Each spell entry encodes the data needed
to compute its damage from build stats: element multipliers, conversions,
spell-vs-melee damage type, and whether attack-speed scaling applies.

Source of truth: wynnbuilder atree.json + damage_calc.js.

Each spell is assigned a stable `spell_id` (the order in SPELLS). The formula
tag stored on a composite is `FORMULA_SPELL_DAMAGE_BASE + spell_id`, so the
search-engine kernel can dispatch to the right spell evaluator without an
extra SoA field.

Indexed access from numba kernels uses the precomputed numpy arrays at the
bottom of this module (SPELL_MULT_*, SPELL_USE_SD, SPELL_IGNORE_SPEED).
"""

import numpy as np


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
#   mults         — (m_n, m_e, m_t, m_w, m_f, m_a) — % multipliers per element
#   use_spell     — True → use sdPct/sdRaw; False → use mdPct/mdRaw
#   ignore_speed  — True → skip baseDamageMultiplier[atkSpd] (DoT/instant)
#   deps          — tuple of stat names in canonical evaluator order

SPELLS = [
    {
        "name": "mage_meteor",
        "class_name": "mage",
        "display": "Meteor",
        "mults": (330, 70, 0, 0, 0, 0),
        "use_spell": True,
        "ignore_speed": False,
        # Order matches _meteor_bounds' dep[] reads — do not shuffle.
        # Note: wynnbuilder has a global `damRaw` stat that we don't carry in
        # our IDS_STATS registry — dropped from deps; the formula accounts for
        # it as 0.
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "eDamPct", "eSdPct", "eSdRaw"),
    },
    {
        "name": "warrior_bash",
        "class_name": "warrior",
        "display": "Bash",
        "mults": (170, 30, 0, 0, 0, 0),
        "use_spell": False,  # melee attack — uses mdPct/mdRaw not sdPct/sdRaw
        "ignore_speed": False,
        # Earth weapon damage = 0 (no eDamRaw stat); earth contribution comes
        # only from eMdRaw raw bonus. Per-element % bonuses on earth (eDamPct)
        # are dead-multiplied (× 0 weapon) so we drop them from deps.
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw", "eMdRaw"),
    },
]


# ============================================================
# Derived numpy arrays (numba-friendly module-level constants)
# ============================================================

SPELL_INDEX = {s["name"]: i for i, s in enumerate(SPELLS)}
SPELL_COUNT = len(SPELLS)

SPELL_MULT_N = np.array([s["mults"][0] for s in SPELLS], dtype=np.int32)
SPELL_MULT_E = np.array([s["mults"][1] for s in SPELLS], dtype=np.int32)
SPELL_MULT_T = np.array([s["mults"][2] for s in SPELLS], dtype=np.int32)
SPELL_MULT_W = np.array([s["mults"][3] for s in SPELLS], dtype=np.int32)
SPELL_MULT_F = np.array([s["mults"][4] for s in SPELLS], dtype=np.int32)
SPELL_MULT_A = np.array([s["mults"][5] for s in SPELLS], dtype=np.int32)

SPELL_USE_SD = np.array([s["use_spell"] for s in SPELLS], dtype=np.bool_)
SPELL_IGNORE_SPEED = np.array([s["ignore_speed"] for s in SPELLS], dtype=np.bool_)


# ============================================================
# Attack speed multipliers (from wynnbuilder build_utils.js)
# ============================================================
# The crafter doesn't know the target weapon's attack speed without context;
# user supplies via _context["atk_speed"]. These multiplier values are applied
# to spell damage when ignore_speed=False.

ATK_SPEED_MULT = {
    "SUPER_SLOW": 0.51,
    "VERY_SLOW": 0.83,
    "SLOW": 1.5,
    "NORMAL": 2.05,
    "FAST": 2.5,
    "VERY_FAST": 3.1,
    "SUPER_FAST": 4.3,
}
