"""
skillpoint_lookup.py

Precomputed lookup tables converting raw skillpoint counts (str/dex/int/def/agi,
0..150) into their derived sub-stat percentages.

Source of truth: wynnbuilder's `js/build_utils.js:8-25`. The base curve is
non-linear (`skillPointsToPercentage`), so per-ingredient additivity is broken:
these tables can only be applied to the build's TOTAL skillpoint count, not to
per-ingredient contributions. Downstream code that wants to score skillpoint
sub-stats has to evaluate at the leaf, on the summed totals.

Two distinct multiplier sets exist in wynnbuilder:
  - `skillpoint_final_mult` = [1, 1, 0.619, 0.867, 0.951] — for each skillpoint's
    HEADLINE effect (str→%dmg, dex→%crit, int→%cost-red, def→%def-red, agi→%dodge)
  - `skillpoint_damage_mult` = [1, 1, 1, 0.867, 0.951] — for each skillpoint's
    ELEMENTAL damage bonus (str→earth, dex→thunder, int→water, def→fire, agi→air)

The two only diverge for int (0.619 vs 1.0). Both tables are exposed.
"""

import numpy as np


# ============================================================
# Index convention — matches wynnbuilder's skp_order
# ============================================================

SKP_NAMES = ("str", "dex", "int", "def", "agi")
SKP_INDEX = {name: i for i, name in enumerate(SKP_NAMES)}

SKP_STR = 0
SKP_DEX = 1
SKP_INT = 2
SKP_DEF = 3
SKP_AGI = 4

# Element mapping: each skillpoint boosts one element's damage.
# str→earth, dex→thunder, int→water, def→fire, agi→air
SKP_ELEMENT_NAMES = ("e", "t", "w", "f", "a")
SKP_ELEMENT_DAM_PCT_STAT = ("eDamPct", "tDamPct", "wDamPct", "fDamPct", "aDamPct")

# Hard cap from wynnbuilder. Beyond this, skillPointsToPercentage clamps.
SKP_MAX = 150


# ============================================================
# Multipliers (verbatim from wynnbuilder)
# ============================================================

# `skillpoint_final_mult` — applied to the headline effect of each skillpoint:
#   str→%damage   dex→%crit   int→%cost-red   def→%def-red   agi→%dodge
# Note: int's 0.619 is `0.5 / skillPointsToPercentage(150)`, so the headline
# tops out at exactly 50% (mana cost reduction is capped at half).
_INT_HEADLINE_MULT = 0.5 / ((0.9908 / (1 - 0.9908) * (1 - 0.9908 ** 150)) / 100.0)

SKP_FINAL_MULT = np.array(
    [1.0, 1.0, _INT_HEADLINE_MULT, 0.867, 0.951], dtype=np.float64,
)

# `skillpoint_damage_mult` — applied to the elemental damage bonus of each
# skillpoint. Same as final_mult except int uses 1.0 here (water damage gets
# the raw curve, not the halved cost-reduction curve).
SKP_DAMAGE_MULT = np.array(
    [1.0, 1.0, 1.0, 0.867, 0.951], dtype=np.float64,
)


# ============================================================
# Base curve: skillPointsToPercentage(skp), 0 ≤ skp ≤ 150
# ============================================================

def _skillpoints_to_percentage(skp: int) -> float:
    """Wynnbuilder's curve. Returns fraction in [0, ~0.8077]."""
    if skp <= 0:
        return 0.0
    if skp >= SKP_MAX:
        skp = SKP_MAX
    r = 0.9908
    return (r / (1 - r) * (1 - r ** skp)) / 100.0


# Shape: (SKP_MAX + 1,). Index by skillpoint count, get base percentage.
SKP_PCT_BASE = np.array(
    [_skillpoints_to_percentage(s) for s in range(SKP_MAX + 1)],
    dtype=np.float64,
)


# ============================================================
# Per-skillpoint lookup tables, shape (5, SKP_MAX + 1)
# ============================================================

# Headline effect (str→%dmg, dex→%crit, int→%cost-red, def→%def-red, agi→%dodge).
# Index as SKP_HEADLINE_PCT[skp_idx, skp_count].
SKP_HEADLINE_PCT = np.outer(SKP_FINAL_MULT, SKP_PCT_BASE)

# Elemental damage % bonus (str→earth, dex→thunder, int→water, def→fire, agi→air).
# Index as SKP_ELEMENT_PCT[skp_idx, skp_count].
SKP_ELEMENT_PCT = np.outer(SKP_DAMAGE_MULT, SKP_PCT_BASE)


# ============================================================
# Convenience: stat-name labels for each sub-stat
# ============================================================

# Display names — useful for downstream UIs and not standardized in our base
# stat registry (only the elemental ones map to existing IDS_STATS keys).
SKP_HEADLINE_LABEL = (
    "strDamPct",      # general %damage from strength (no IDS_STATS equivalent)
    "critPct",        # crit chance % (no IDS_STATS equivalent)
    "costRedPct",     # mana cost reduction % (no IDS_STATS equivalent)
    "defRedPct",      # damage reduction % (no IDS_STATS equivalent)
    "dodgePct",       # dodge chance % (no IDS_STATS equivalent)
)


def lookup_headline(skp_idx: int, skp_count: int) -> float:
    """Headline sub-stat percentage for a given skillpoint count. Clamps to [0, 150]."""
    if skp_count < 0:
        return 0.0
    if skp_count > SKP_MAX:
        skp_count = SKP_MAX
    return float(SKP_HEADLINE_PCT[skp_idx, skp_count])


def lookup_element(skp_idx: int, skp_count: int) -> float:
    """Elemental damage % for a given skillpoint count. Clamps to [0, 150]."""
    if skp_count < 0:
        return 0.0
    if skp_count > SKP_MAX:
        skp_count = SKP_MAX
    return float(SKP_ELEMENT_PCT[skp_idx, skp_count])
