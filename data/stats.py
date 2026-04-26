"""
stats.py

Global stat registry.

This file defines every stat supported by the crafted build generator.
All stats are flattened into a single dense index space for performance.

Every stat can have:
- min
- max
- weight

Query system uses this registry to map stat names -> indices.
"""


# ============================================================
# IDENTIFICATION STATS (ids)
# ============================================================

IDS_STATS = tuple(sorted([
    "aDamPct", "aDamRaw", "aDefPct", "aMdRaw", "aSdPct", "aSdRaw",
    "agi",
    "atkTier",
    "damPct",
    "def",
    "dex",
    "eDamPct", "eDamRaw", "eDefPct", "eMdRaw", "eSdPct", "eSdRaw", "eSteal",
    "expd",
    "fDamPct", "fDamRaw", "fDefPct", "fMdRaw", "fSdPct", "fSdRaw",
    "gSpd", "gXp",
    "healPct",
    "hpBonus",
    "hprPct", "hprRaw",
    "int",
    "jh",
    "kb",
    "lb",
    "lq",
    "ls",
    "maxMana",
    "mdPct", "mdRaw",
    "mr",
    "ms",
    "nDamPct", "nDamRaw", "nMdRaw",
    "poison",
    "rDamPct", "rDefPct", "rSdPct",
    "ref",
    "sdPct", "sdRaw",
    "spd",
    "sprint", "sprintReg",
    "str",
    "tDamPct", "tDamRaw", "tDefPct", "tMdRaw", "tSdRaw",
    "thorns",
    "wDamPct", "wDamRaw", "wDefPct", "wMdRaw", "wSdPct", "wSdRaw",
    "xpb"
]))


# ============================================================
# REQUIREMENT STATS (itemIDs, except dura)
# ============================================================

REQ_STATS = (
    "strReq",
    "dexReq",
    "intReq",
    "defReq",
    "agiReq",
)


# ============================================================
# SPECIAL STATS
# ============================================================

# Durability for crafted equipment
STAT_DURABILITY = "durability"

# Duration for consumables
STAT_DURATION = "duration"

# Consumable charges
STAT_CHARGES = "charges"

SPECIAL_STATS = (
    STAT_DURABILITY,
    STAT_DURATION,
    STAT_CHARGES,
)

# ============================================================
# DERIVED STATS
# ============================================================

DERIVED_STATS = (
    "hpr",
    "ehp",
    "ehpr",
    # Spells (see data/spells.py for catalog order — must match SPELL ids)
    "mage_meteor",
    "mage_ice_snake",
    "warrior_bash",
    "warrior_uppercut",
    "warrior_war_scream",
    "archer_arrow_storm",
    "archer_arrow_bomb",
    "assassin_spin_attack",
    "assassin_multihit",
    "assassin_smoke_bomb",
    "shaman_totem",
    "shaman_aura",
    "shaman_uproot",
)

# Spell composites: dep order is the canonical layout defined in data/spells.py
# (SPELL_SPELL_DEPS for use_spell=True, SPELL_MELEE_DEPS otherwise). Mirrored
# below so the query parser doesn't need to import that module — but the lists
# MUST match `data/spells.py` (a defensive assert in the parser would catch drift).
_SPELL_DEPS = (
    "nDamRaw", "eDamRaw", "tDamRaw", "wDamRaw", "fDamRaw", "aDamRaw",
    "nDamPct", "eDamPct", "tDamPct", "wDamPct", "fDamPct", "aDamPct",
    "eSdPct", "wSdPct", "fSdPct", "aSdPct",
    "eSdRaw", "tSdRaw", "wSdRaw", "fSdRaw", "aSdRaw",
    "sdPct", "damPct", "sdRaw",
    "rDamPct", "rSdPct",
    "str", "dex", "int", "def", "agi",
)

_MELEE_DEPS = (
    "nDamRaw", "eDamRaw", "tDamRaw", "wDamRaw", "fDamRaw", "aDamRaw",
    "nDamPct", "eDamPct", "tDamPct", "wDamPct", "fDamPct", "aDamPct",
    "nMdRaw", "eMdRaw", "tMdRaw", "wMdRaw", "fMdRaw", "aMdRaw",
    "mdPct", "damPct", "mdRaw",
    "rDamPct",
    "str", "dex", "int", "def", "agi",
)

DERIVED_DEPENDENCIES = {
    "hpr": ("hprRaw", "hprPct"),
    "ehp": ("hpBonus", "def", "agi"),
    "ehpr": ("hprRaw", "hprPct", "def", "agi"),
    # All spell composites use one of the two canonical layouts above.
    "mage_meteor":          _SPELL_DEPS,
    "mage_ice_snake":       _SPELL_DEPS,
    "warrior_bash":         _MELEE_DEPS,
    "warrior_uppercut":     _MELEE_DEPS,
    "warrior_war_scream":   _MELEE_DEPS,
    "archer_arrow_storm":   _SPELL_DEPS,
    "archer_arrow_bomb":    _SPELL_DEPS,
    "assassin_spin_attack": _MELEE_DEPS,
    "assassin_multihit":    _MELEE_DEPS,
    "assassin_smoke_bomb":  _SPELL_DEPS,
    "shaman_totem":         _SPELL_DEPS,
    "shaman_aura":          _SPELL_DEPS,
    "shaman_uproot":        _MELEE_DEPS,
}

# Formula tag per derived stat. Mapped to int in query.py for numba passage.
# Supported formulas:
#   "mul_div_100" (arity 2): a * b // 100
#   "raw_to_pct"  (arity 2): wynnbuilder's rawToPct(raw, pct_delta) — sign-asymmetric
#                   raw>0 → (raw * (100 + delta)) // 100
#                   raw<0 → min(0, (raw * (100 - delta)) // 100)
#                   raw=0 → 0
#   "ehp"         (arity 3): hpBonus / (1 - (1 - agi_pct) * def_pct)
#                   Simplified from wynnbuilder's getDefenseStats — ignores
#                   level-base hp, classDef, defMult, agiDef (assumed 0/1/0).
#                   def_pct, agi_pct from SKP_HEADLINE_PCT lookup, clamped 0..150.
#   "ehpr"        (arity 4): rawToPct(hprRaw, hprPct) / (1 - (1 - agi_pct) * def_pct)
#                   Same simplifications as ehp.
#   ("spell", N)  (variable arity): per-spell damage formula, where N is the
#                 spell_id from data.spells.SPELL_INDEX. The formula tag stored
#                 on the composite is FORMULA_SPELL_DAMAGE_BASE + N. Build-context
#                 (skillpoints, atk_speed, crit) supplied via _context in query.
DERIVED_FORMULA = {
    "hpr": "raw_to_pct",
    "ehp": "ehp",
    "ehpr": "ehpr",
    # Spell IDs match SPELLS list in data/spells.py.
    "mage_meteor":          ("spell", 0),
    "mage_ice_snake":       ("spell", 1),
    "warrior_bash":         ("spell", 2),
    "warrior_uppercut":     ("spell", 3),
    "warrior_war_scream":   ("spell", 4),
    "archer_arrow_storm":   ("spell", 5),
    "archer_arrow_bomb":    ("spell", 6),
    "assassin_spin_attack": ("spell", 7),
    "assassin_multihit":    ("spell", 8),
    "assassin_smoke_bomb":  ("spell", 9),
    "shaman_totem":         ("spell", 10),
    "shaman_aura":          ("spell", 11),
    "shaman_uproot":        ("spell", 12),
}

# Stats whose "empty" value is non-zero (i.e. a fresh item already has them).
# Applied to base_min_stats / base_max_stats when the stat becomes active,
# unless the user passes an explicit "base" / "min_base" / "max_base".
DEFAULT_BASE = {}

# Numba-compatible formula int tags. Keep in sync with DERIVED_FORMULA values.
FORMULA_MUL_DIV_100 = 0
FORMULA_RAW_TO_PCT = 1
FORMULA_EHP = 2
FORMULA_EHPR = 3
# Spell formulas: tag = FORMULA_SPELL_DAMAGE_BASE + spell_id. Choose a value
# well above the fixed-tag range so kernels can branch via `f >= base`.
FORMULA_SPELL_DAMAGE_BASE = 100

# ============================================================
# CONSU SKILLS
# ============================================================

CONSU_SKILLS = ["ALCHEMISM", "SCRIBING", "COOKING"]


# ============================================================
# GLOBAL STAT LIST
# ============================================================

ALL_STATS = IDS_STATS + REQ_STATS + SPECIAL_STATS

STAT_COUNT = len(ALL_STATS)

# Fast name -> index lookup (only used during load/query parsing).
# Derived from ALL_STATS so adding a stat to IDS_STATS automatically reindexes
# the rest — no manual table to keep in sync.
STAT_INDEX = {name: i for i, name in enumerate(ALL_STATS)}

REQ_STATS_IDX = tuple(STAT_INDEX[name] for name in REQ_STATS)


# ============================================================
# HELPER CONSTANTS
# ============================================================

IDX_DURABILITY = STAT_INDEX[STAT_DURABILITY]
IDX_DURATION = STAT_INDEX[STAT_DURATION]
IDX_CHARGES = STAT_INDEX[STAT_CHARGES]