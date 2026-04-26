import math
import numpy as np

from typing import NamedTuple
from data.stats import STAT_INDEX, IDX_DURABILITY, IDX_DURATION, IDX_CHARGES


TIER_MULT = (0.0, 1.0, 1.25, 1.4)


class Recipe(NamedTuple):

    base_min_stats_proj: np.ndarray
    base_max_stats_proj: np.ndarray

    scaled_dura_min: int
    scaled_dura_max: int

    # Crafted weapon's intrinsic neutral damage range (post-mat-mult,
    # pre-powder, pre-roll). For non-weapons this is None. The search
    # pipeline overlays this onto `query.build_ctx[BUILD_CTX_WD_N]` so the
    # spell kernel scales it by m_n / m_e (instead of treating the raw
    # number as a gear ID raw, which was the previous over-counting bug).
    weapon_dam_neutral: tuple = (0, 0)


def build_recipe(raw_recipe: dict, query, tier: int) -> Recipe:

    recipe_data = raw_recipe.data

    # ------------------------------------------------------------
    # Base durability from recipe
    # ------------------------------------------------------------
    if not query.consumable:
        base_dura_min = recipe_data["durability"]["minimum"]
        base_dura_max = recipe_data["durability"]["maximum"]
    else:
        base_dura_min = recipe_data["duration"]["minimum"]
        base_dura_max = recipe_data["duration"]["maximum"]

    mult = TIER_MULT[tier]

    scaled_dura_min = math.floor(base_dura_min * mult)
    scaled_dura_max = math.floor(base_dura_max * mult)

    # ------------------------------------------------------------
    # Start from query projected base constraints
    # ------------------------------------------------------------

    base_min_stats_proj = query.base_min_stats_proj.copy()
    base_max_stats_proj = query.base_max_stats_proj.copy()

    # ------------------------------------------------------------
    # Inject dura and consu stats
    # ------------------------------------------------------------

    proj_map = query.proj_stats_idx

    def inject_stat(stat_idx_full, vmin, vmax):
        """Inject a stat into projected base vectors."""
        p = proj_map[stat_idx_full]
        if p == -1:
            return
    
        if base_min_stats_proj[p] == 0:
            base_min_stats_proj[p] = vmin
        else:
            base_min_stats_proj[p] = max(base_min_stats_proj[p], vmin)
    
        if base_max_stats_proj[p] == 0:
            base_max_stats_proj[p] = vmax
        else:
            base_max_stats_proj[p] = min(base_max_stats_proj[p], vmax)
            
            
    if query.consumable:
        inject_stat(IDX_DURATION, scaled_dura_min, scaled_dura_max)
    else:
        inject_stat(IDX_DURABILITY, scaled_dura_min, scaled_dura_max)
        
    if "basicDuration" in recipe_data:
        charges_min = recipe_data["basicDuration"]["minimum"]
        charges_max = recipe_data["basicDuration"]["maximum"]
    
        inject_stat(IDX_CHARGES, charges_min, charges_max)
        
    weapon_dam_neutral = (0, 0)
    if not query.consumable and "healthOrDamage" in recipe_data:

        hod_min = recipe_data["healthOrDamage"]["minimum"]
        hod_max = recipe_data["healthOrDamage"]["maximum"]

        if query.skill in ("TAILORING", "ARMOURING"):
            # Armor: healthOrDamage maps to hpBonus (stat used for ehp etc.).
            inject_stat(STAT_INDEX["hpBonus"], hod_min, hod_max)
        else:
            # Weapons: don't inject into nDamRaw — the spell kernel reads
            # weapon damage from ctx.weapon_dam (set by search_pipelined
            # from this `weapon_dam_neutral` field). Folding hod into
            # nDamRaw caused ingredient nDamRaw rolls to be incorrectly
            # treated as weapon damage (multiplied by m_n + boost_t),
            # ~6× over-valuing per-element raws vs wynnbuilder.
            weapon_dam_neutral = (int(hod_min), int(hod_max))

    # ------------------------------------------------------------

    return Recipe(
        base_min_stats_proj=base_min_stats_proj,
        base_max_stats_proj=base_max_stats_proj,
        scaled_dura_min=scaled_dura_min,
        scaled_dura_max=scaled_dura_max,
        weapon_dam_neutral=weapon_dam_neutral,
    )

















