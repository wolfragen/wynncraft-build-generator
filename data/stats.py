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
    "aDamPct", "aDamRaw", "aDefPct", "aSdPct",
    "agi",
    "atkTier",
    "damPct",
    "def",
    "dex",
    "eDamPct", "eDefPct", "eMdRaw", "eSdPct", "eSdRaw", "eSteal",
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
    "nDamRaw", "nMdRaw",
    "poison",
    "rDamPct", "rDefPct",
    "ref",
    "sdPct", "sdRaw",
    "spd",
    "sprint", "sprintReg",
    "str",
    "tDamPct", "tDamRaw", "tDefPct", "tMdRaw",
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
    "hprEff",
)

DERIVED_DEPENDENCIES = {
    "hprEff": ("hprRaw", "hprPct"),
}

# Formula tag per derived stat. Mapped to int in query.py for numba passage.
# Supported formulas:
#   "mul_div_100" : a * b // 100  (exactly two deps)
DERIVED_FORMULA = {
    "hprEff": "mul_div_100",
}

# Stats whose "empty" value is non-zero (i.e. a fresh item already has them).
# Applied to base_min_stats / base_max_stats when the stat becomes active,
# unless the user passes an explicit "base" / "min_base" / "max_base".
DEFAULT_BASE = {
    "hprPct": 100,
}

# Numba-compatible formula int tags. Keep in sync with DERIVED_FORMULA values.
FORMULA_MUL_DIV_100 = 0

# ============================================================
# CONSU SKILLS
# ============================================================

CONSU_SKILLS = ["ALCHEMISM", "SCRIBING", "COOKING"]


# ============================================================
# GLOBAL STAT LIST
# ============================================================

ALL_STATS = IDS_STATS + REQ_STATS + SPECIAL_STATS

STAT_COUNT = len(ALL_STATS)

# Fast name -> index lookup (only used during load/query parsing)
STAT_INDEX = {
    'aDamPct': 0,
    'aDamRaw': 1,
    'aDefPct': 2,
    'aSdPct': 3,
    'agi': 4,
    'atkTier': 5,
    'damPct': 6,
    'def': 7,
    'dex': 8,
    'eDamPct': 9,
    'eDefPct': 10,
    'eMdRaw': 11,
    'eSdPct': 12,
    'eSdRaw': 13,
    'eSteal': 14,
    'expd': 15,
    'fDamPct': 16,
    'fDamRaw': 17,
    'fDefPct': 18,
    'fMdRaw': 19,
    'fSdPct': 20,
    'fSdRaw': 21,
    'gSpd': 22,
    'gXp': 23,
    'healPct': 24,
    'hpBonus': 25,
    'hprPct': 26,
    'hprRaw': 27,
    'int': 28,
    'jh': 29,
    'kb': 30,
    'lb': 31,
    'lq': 32,
    'ls': 33,
    'maxMana': 34,
    'mdPct': 35,
    'mdRaw': 36,
    'mr': 37,
    'ms': 38,
    'nDamRaw': 39,
    'nMdRaw': 40,
    'poison': 41,
    'rDamPct': 42,
    'rDefPct': 43,
    'ref': 44,
    'sdPct': 45,
    'sdRaw': 46,
    'spd': 47,
    'sprint': 48,
    'sprintReg': 49,
    'str': 50,
    'tDamPct': 51,
    'tDamRaw': 52,
    'tDefPct': 53,
    'tMdRaw': 54,
    'thorns': 55,
    'wDamPct': 56,
    'wDamRaw': 57,
    'wDefPct': 58,
    'wMdRaw': 59,
    'wSdPct': 60,
    'wSdRaw': 61,
    'xpb': 62,
    'strReq': 63,
    'dexReq': 64,
    'intReq': 65,
    'defReq': 66,
    'agiReq': 67,
    'durability': 68,
    'duration': 69,
    'charges': 70,
}

REQ_STATS_IDX = (63, 64, 65, 66, 67)


# ============================================================
# HELPER CONSTANTS
# ============================================================

IDX_DURABILITY = STAT_INDEX[STAT_DURABILITY]
IDX_DURATION = STAT_INDEX[STAT_DURATION]
IDX_CHARGES = STAT_INDEX[STAT_CHARGES]