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
    # ---- Mage ----
    {
        "name": "mage_meteor", "class_name": "mage", "display": "Meteor",
        "mults": (330, 70, 0, 0, 0, 0), "use_spell": True, "ignore_speed": False,
        # Earth has no eDamRaw → eDamPct/eSdPct still in deps (dead-multiplied
        # but kept for legacy reasons; bound function handles them correctly).
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "eDamPct", "eSdPct", "eSdRaw"),
    },
    {
        "name": "mage_ice_snake", "class_name": "mage", "display": "Ice Snake",
        "mults": (120, 0, 0, 55, 0, 0), "use_spell": True, "ignore_speed": False,
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "wDamRaw", "wDamPct", "wSdPct", "wSdRaw"),
    },
    # ---- Warrior ----
    {
        "name": "warrior_bash", "class_name": "warrior", "display": "Bash",
        "mults": (170, 30, 0, 0, 0, 0), "use_spell": False, "ignore_speed": False,
        # Earth has no eDamRaw → element pcts dropped (dead).
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw", "eMdRaw"),
    },
    {
        "name": "warrior_uppercut", "class_name": "warrior", "display": "Uppercut",
        "mults": (260, 40, 40, 0, 0, 0), "use_spell": False, "ignore_speed": False,
        # 3-elem: neutral + earth + thunder. Earth weapon=0 → eDamPct dropped.
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw", "eMdRaw",
                 "tDamRaw", "tDamPct", "tMdRaw"),
    },
    {
        "name": "warrior_war_scream", "class_name": "warrior", "display": "War Scream",
        "mults": (75, 0, 0, 0, 25, 0), "use_spell": False, "ignore_speed": False,
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw",
                 "fDamRaw", "fDamPct", "fMdRaw"),
    },
    # ---- Archer ----
    {
        "name": "archer_arrow_storm", "class_name": "archer", "display": "Arrow Storm",
        "mults": (30, 0, 10, 0, 0, 0), "use_spell": True, "ignore_speed": False,
        # Thunder has tDamRaw + tDamPct, no tSdPct or tSdRaw.
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "tDamRaw", "tDamPct"),
    },
    {
        "name": "archer_arrow_bomb", "class_name": "archer", "display": "Arrow Bomb",
        "mults": (140, 0, 0, 0, 20, 0), "use_spell": True, "ignore_speed": False,
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "fDamRaw", "fDamPct", "fSdPct", "fSdRaw"),
    },
    # ---- Assassin ----
    {
        "name": "assassin_spin_attack", "class_name": "assassin", "display": "Spin Attack",
        "mults": (120, 0, 30, 0, 0, 0), "use_spell": False, "ignore_speed": False,
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw",
                 "tDamRaw", "tDamPct", "tMdRaw"),
    },
    {
        "name": "assassin_multihit", "class_name": "assassin", "display": "Multihit",
        "mults": (30, 0, 0, 10, 0, 0), "use_spell": False, "ignore_speed": False,
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw",
                 "wDamRaw", "wDamPct", "wMdRaw"),
    },
    {
        "name": "assassin_smoke_bomb", "class_name": "assassin", "display": "Smoke Bomb",
        "mults": (25, 5, 0, 0, 0, 5), "use_spell": True, "ignore_speed": True,  # DoT
        # 3-elem: n + e + a. Earth weapon=0 → eDamPct/eSdPct dropped.
        # Air has no aSdRaw → only aDamPct + aSdPct.
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct", "eSdRaw",
                 "aDamRaw", "aDamPct", "aSdPct"),
    },
    # ---- Shaman ----
    {
        "name": "shaman_totem", "class_name": "shaman", "display": "Totem (DPS)",
        "mults": (6, 0, 0, 0, 0, 6), "use_spell": True, "ignore_speed": True,  # DoT
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "aDamRaw", "aDamPct", "aSdPct"),
    },
    {
        "name": "shaman_aura", "class_name": "shaman", "display": "Aura",
        "mults": (150, 0, 0, 30, 0, 0), "use_spell": True, "ignore_speed": False,
        "deps": ("nDamRaw", "sdPct", "sdRaw", "damPct",
                 "wDamRaw", "wDamPct", "wSdPct", "wSdRaw"),
    },
    {
        "name": "shaman_uproot", "class_name": "shaman", "display": "Uproot",
        "mults": (80, 30, 20, 0, 0, 0), "use_spell": False, "ignore_speed": False,
        # 3-elem: n + e + t. Earth weapon=0 → eDamPct dropped.
        "deps": ("nDamRaw", "mdPct", "mdRaw", "damPct", "nMdRaw", "eMdRaw",
                 "tDamRaw", "tDamPct", "tMdRaw"),
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
