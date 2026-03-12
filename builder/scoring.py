from numpy import clip as clamp
from data_loader import stat_to_max
from models import BuildInfo

def score_build_spellDmg(build_info: BuildInfo):
    spellRaw = build_info.stats.get("sdRaw", 0)
    spellPct = build_info.stats.get("sdPct", 0)
    dps = build_info.stats.get("averageDps", 0)
    skStr = build_info.skill_points["str"]["attributed"] + build_info.skill_points["str"]["items_counting"] + build_info.skill_points["str"]["items_not_counting"]
    return max(0, dps + spellRaw) * (1 + spellPct / 100) * (1 + clamp(skStr, 0, 150) * (0.7 / 150))

def score_build_ehp(build_info: BuildInfo):
    hp = build_info.stats.get("hp", 0)
    hp += build_info.stats.get("hpBonus", 0)
    skDef = build_info.skill_points["def"]["attributed"] + build_info.skill_points["def"]["items_counting"] + build_info.skill_points["def"]["items_not_counting"]
    skAgi = build_info.skill_points["agi"]["attributed"] + build_info.skill_points["agi"]["items_counting"] + build_info.skill_points["agi"]["items_not_counting"]
    return hp * (1 + clamp(skDef, 0, 150) * (0.7 / 150)) * (1 + clamp(skAgi, 0, 150) * (0.7 / 150))

def score_build_raw_hp_hpr_thorns_reflex(build_info: BuildInfo):
    hpWeight = 1
    hprWeight = 8

    hp = build_info.stats.get("hp", 0)
    hp += build_info.stats.get("hpBonus", 0)
    hprRaw = build_info.stats.get("hprRaw", 0)
    hrpPct = build_info.stats.get("hprPct", 0)
    trueHpr = hprRaw * (1 + hrpPct / 100)
    thorns = build_info.stats.get("thorns", 0)
    reflexion = build_info.stats.get("ref", 0)
    
    if thorns < 100 or reflexion < 100:
        return 0
    else:
        return hp * hpWeight + trueHpr * hprWeight

def score_item_spellDmg(item: dict, partial_build_info: BuildInfo):
    spellRawBuild = partial_build_info.stats.get("sdRaw", 0)
    spellRawItem = stat_to_max(item.get("sdRaw", 0))
    spellPctBuild = partial_build_info.stats.get("sdPct", 0)
    spellPctItem = stat_to_max(item.get("sdPct", 0))
    skStrBuild = partial_build_info.skill_points["str"]["attributed"] + partial_build_info.skill_points["str"]["items_counting"] + partial_build_info.skill_points["str"]["items_not_counting"]
    skStrItem = stat_to_max(item.get("str", 0))
    
    if item.get("category") == "weapon":
        dps = item.get("averageDps", 0)
    else:
        dps = partial_build_info.stats.get("averageDps", 0)
        
    return max(0, dps + spellRawBuild + spellRawItem) * (1 + (spellPctBuild + spellPctItem) / 100) * (1 + clamp(skStrBuild + skStrItem, 0, 150) * (0.7 / 150))

def score_item_ehp(item: dict, partial_build_info: BuildInfo):
    hpBuild = partial_build_info.stats.get("hp", 0)
    hpBuild += partial_build_info.stats.get("hpBonus", 0)
    hpItem = stat_to_max(item.get("hp", 0))
    hpItem += stat_to_max(item.get("hpBonus", 0))
    
    skDefBuild = partial_build_info.skill_points["def"]["attributed"] + partial_build_info.skill_points["def"]["items_counting"] + partial_build_info.skill_points["def"]["items_not_counting"]
    skDefItem = stat_to_max(item.get("def", 0))
    skAgiBuild = partial_build_info.skill_points["agi"]["attributed"] + partial_build_info.skill_points["agi"]["items_counting"] + partial_build_info.skill_points["agi"]["items_not_counting"]
    skAgiItem = stat_to_max(item.get("agi", 0))
    
    return (hpBuild + hpItem) * (1 + clamp(skDefBuild + skDefItem, 0, 150) * (0.7 / 150)) * (1 + clamp(skAgiBuild + skAgiItem, 0, 150) * (0.7 / 150))

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
    trueHpr = (hprRawBuild + hprRawItem) * (1 + (hrpPctBuild + hrpPctItem) / 100)

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
