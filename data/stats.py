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

# Sorted deterministically to guarantee stable indexing
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
# GLOBAL STAT LIST
# ============================================================

ALL_STATS = IDS_STATS + REQ_STATS + SPECIAL_STATS

STAT_COUNT = len(ALL_STATS)

# Fast name -> index lookup (only used during load/query parsing)
STAT_INDEX = {name: i for i, name in enumerate(ALL_STATS)}


# ============================================================
# HELPER CONSTANTS
# ============================================================

IDX_DURABILITY = STAT_INDEX[STAT_DURABILITY]
IDX_DURATION = STAT_INDEX[STAT_DURATION]
IDX_CHARGES = STAT_INDEX[STAT_CHARGES]