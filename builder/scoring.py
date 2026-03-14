import os
import csv
from numpy import clip as clamp
from data_loader import stat_to_max
from models import BuildInfo

# Load skill point bonuses
# Wiki Info:
# - Max manual allocation: 100. Final max with items: 150 (diminishing returns).
# - Negative points = 0. Above 150 = same as 150.
# - Str/Dex: Same scale (0% at 0 pts to 80.8% at 150 pts).
# - Int: Water dmg & Max mana bonus use the same scale (0-80.8%).
# - Int Spell Cost Reduction: (0.5 / (80.8 / ScaledIntelligence)) where ScaledIntelligence is the scale value.
# - Def: Multiplies Str/Dex scale by 0.867.
# - Agi: Multiplies Str/Dex scale by 0.951.
# - Rounding: In-game values rounded to nearest tenth, but precise values used for calculations.
SP_BONUS_TABLE = [0.0] * 151
try:
    csv_path = os.path.join(os.path.dirname(__file__), 'skillpoints.csv')
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                assigned = int(row['assigned'])
                if 0 <= assigned <= 150:
                    # Use 'calculated' for precise damage calculation
                    SP_BONUS_TABLE[assigned] = float(row['calculated'])
except Exception as e:
    print(f"Warning: Could not load skillpoints.csv: {e}")

def get_sp_scale(pts):
    """Returns the base 0-0.808 scale value used by all stats."""
    pts_int = int(clamp(pts, 0, 150))
    return SP_BONUS_TABLE[pts_int]

def get_str_dex_bonus(pts):
    return get_sp_scale(pts)

def get_int_bonus(pts):
    return get_sp_scale(pts)

def get_spell_cost_reduction(pts):
    # (0.5 / (0.808 / scale)) = 0.5 * (scale / 0.808)
    scale = get_sp_scale(pts)
    return 0.5 * (scale / 0.8077572218478628) # Using precise 150 value

def get_def_bonus(pts):
    return get_sp_scale(pts) * 0.867

def get_agi_bonus(pts):
    return get_sp_scale(pts) * 0.951

# Wynncraft constants
ATTACK_SPEEDS = ["SUPER_SLOW", "VERY_SLOW", "SLOW", "NORMAL", "FAST", "VERY_FAST", "SUPER_FAST"]
SPEED_MULTIPLIERS = [0.51, 0.83, 1.5, 2.05, 2.5, 3.1, 4.3]

DAMAGE_KEYS = {
    'nDam': 'n',
    'eDam': 'e',
    'tDam': 't',
    'wDam': 'w',
    'fDam': 'f',
    'aDam': 'a'
}

def parse_range(r):
    if isinstance(r, str) and '-' in r:
        try:
            low, high = r.split('-')
            return (float(low) + float(high)) / 2
        except ValueError:
            return 0.0
    return float(r) if r is not None else 0.0

def get_weapon_base_damage(weapon):
    return {k: parse_range(weapon.get(k, 0)) for k in DAMAGE_KEYS.keys()}

def calculate_full_spell_dmg(stats, weapon):
    if not weapon:
        return 0
    
    base_damages = get_weapon_base_damage(weapon)
    total_base = sum(base_damages.values())
    
    # Speed Multiplier
    speed = weapon.get("atkSpd", "NORMAL")
    try:
        speed_mult = SPEED_MULTIPLIERS[ATTACK_SPEEDS.index(speed)]
    except ValueError:
        speed_mult = 2.05
        
    # Skill Points
    skStr = stats.get("str", 0)
    skDex = stats.get("dex", 0)
    skInt = stats.get("int", 0)
    skDef = stats.get("def", 0)
    skAgi = stats.get("agi", 0)
    
    # Strength Multiplier (Total Damage %)
    str_mult = 1 + get_str_dex_bonus(skStr)
    
    # Dexterity (Crit) Multiplier
    crit_chance = get_str_dex_bonus(skDex)
    crit_mult = 1 + crit_chance * 0.5 # Crit is +50% damage
    
    # Global Multipliers
    sdPct = stats.get("sdPct", 0)
    damPct = stats.get("damPct", 0)
    global_pct = (sdPct + damPct) / 100
    
    # Raw Bonuses
    sdRaw = stats.get("sdRaw", 0)
    damRaw = stats.get("damRaw", 0)
    total_raw = sdRaw + damRaw
    
    # Elemental Bonuses (Items + Skill Points)
    ele_pcts = {
        'eDam': (stats.get("eDamPct", 0) / 100) + get_str_dex_bonus(skStr),
        'tDam': (stats.get("tDamPct", 0) / 100) + get_str_dex_bonus(skDex),
        'wDam': (stats.get("wDamPct", 0) / 100) + get_int_bonus(skInt),
        'fDam': (stats.get("fDamPct", 0) / 100) + get_def_bonus(skDef),
        'aDam': (stats.get("aDamPct", 0) / 100) + get_agi_bonus(skAgi),
        'nDam': 0
    }
    
    total_damage = 0
    
    # Proportional Raw Damage distribution
    if total_base > 0:
        for k, base in base_damages.items():
            ele_raw = total_raw * (base / total_base)
            total_damage += max(0, base + ele_raw) * max(0, 1 + global_pct + ele_pcts[k])
    else:
        # If no base damage, add all raw to neutral
        total_damage += max(0, total_raw) * max(0, 1 + global_pct)
    
    # Apply Strength, Crit, and Speed
    final_damage = total_damage * str_mult * crit_mult * speed_mult
    
    return max(0, final_damage)

def score_build_spellDmg(build_info: BuildInfo):
    weapon = build_info.items.get(build_info.weapon_type)
    return calculate_full_spell_dmg(build_info.stats, weapon)

def score_build_ehp(build_info: BuildInfo):
    hp = build_info.stats.get("hp", 0)
    hp += build_info.stats.get("hpBonus", 0)
    skDef = build_info.stats.get("def", 0)
    skAgi = build_info.stats.get("agi", 0)
    return hp * (1 + get_def_bonus(skDef)) * (1 + get_agi_bonus(skAgi))

def score_build_raw_hp_hpr_thorns_reflex(build_info: BuildInfo):
    hpWeight = 1
    hprWeight = 8

    hp = build_info.stats.get("hp", 0)
    hp += build_info.stats.get("hpBonus", 0)
    hprRaw = build_info.stats.get("hprRaw", 0)
    hrpPct = build_info.stats.get("hprPct", 0)
    trueHpr = hprRaw * max(0, 1 + hrpPct / 100)
    thorns = build_info.stats.get("thorns", 0)
    reflexion = build_info.stats.get("ref", 0)
    
    if thorns < 100 or reflexion < 100:
        return 0
    else:
        return hp * hpWeight + trueHpr * hprWeight

def get_combined_stats(item, partial_build_info):
    stats = partial_build_info.stats.copy()
    for k, v in item.items():
        if k in skill_point_types:
            continue # Already handled in BuildInfo or separately
        if k in maximized_stats:
            stats[k] = max(stats.get(k, 0), stat_to_max(v))
        elif k in build_unique_stats:
            pass # Weapon stats handled separately
        elif k not in item_only_stats:
            stats[k] = stats.get(k, 0) + stat_to_max(v)
    
    # Add skill points from item
    for sk in skill_point_types:
        if item.get("category") != "weapon":
            stats[sk] += stat_to_max(item.get(sk, 0))
            
    return stats

# Need to import these for get_combined_stats to work if I move it inside
from data_loader import skill_point_types, maximized_stats, build_unique_stats, item_only_stats

def score_item_spellDmg(item: dict, partial_build_info: BuildInfo):
    # Combine stats
    stats = partial_build_info.stats.copy()
    for k, v in item.items():
        if k in skill_point_types:
            continue
        if k in maximized_stats:
            stats[k] = max(stats.get(k, 0), stat_to_max(v))
        elif k in build_unique_stats:
            pass
        elif k not in item_only_stats:
            stats[k] = stats.get(k, 0) + stat_to_max(v)
    
    # Add skill points from item
    for sk in skill_point_types:
        if item.get("category") != "weapon":
            stats[sk] += stat_to_max(item.get(sk, 0))
            
    if item.get("category") == "weapon":
        weapon = item
    else:
        weapon = partial_build_info.items.get(partial_build_info.weapon_type)
        
    return calculate_full_spell_dmg(stats, weapon)

def score_item_ehp(item: dict, partial_build_info: BuildInfo):
    hpBuild = partial_build_info.stats.get("hp", 0)
    hpBuild += partial_build_info.stats.get("hpBonus", 0)
    hpItem = stat_to_max(item.get("hp", 0))
    hpItem += stat_to_max(item.get("hpBonus", 0))
    
    skDefBuild = partial_build_info.stats.get("def", 0)
    skDefItem = stat_to_max(item.get("def", 0))
    skAgiBuild = partial_build_info.stats.get("agi", 0)
    skAgiItem = stat_to_max(item.get("agi", 0))
    
    return (hpBuild + hpItem) * (1 + get_def_bonus(skDefBuild + skDefItem)) * (1 + get_agi_bonus(skAgiBuild + skAgiItem))

def score_item_raw_hp_hpr_thorns_reflex(item: dict, partial_build_info: BuildInfo):
    hpWeight = 2
    hprWeight = 16
    thornsWeight = 8
    reflexionWeight = 8

    hpBuild = partial_build_info.stats.get("hp", 0)
    hpBuild += partial_build_info.stats.get("hpBonus", 0)
    hpItem = stat_to_max(item.get("hp", 0))
    hpItem += stat_to_max(item.get("hpBonus", 0))
    
    hprRawBuild = partial_build_info.stats.get("hprRaw", 0)
    hprRawItem = stat_to_max(item.get("hprRaw", 0))
    hrpPctBuild = partial_build_info.stats.get("hprPct", 0)
    hrpPctItem = stat_to_max(item.get("hprPct", 0))
    trueHpr = (hprRawBuild + hprRawItem) * max(0, 1 + (hrpPctBuild + hrpPctItem) / 100)

    thornsBuild = partial_build_info.stats.get("thorns", 0)
    thornsItem = stat_to_max(item.get("thorns", 0))
    reflexionBuild = partial_build_info.stats.get("ref", 0)
    reflexionItem = stat_to_max(item.get("ref", 0))
    
    if thornsBuild + thornsItem > 100:
        thornsWeight = 0
    if reflexionBuild + reflexionItem > 100:
        reflexionWeight = 0
    
    return (hpBuild + hpItem) * hpWeight + trueHpr * hprWeight + (thornsBuild + thornsItem) * thornsWeight + (reflexionBuild + reflexionItem) * reflexionWeight

def score_build_custom(build_info: BuildInfo):
    return 0

def score_item_custom(item: dict, partial_build_info: BuildInfo):
    return 0
