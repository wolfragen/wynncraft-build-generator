import json

skill_point_types = ["str", "dex", "int", "def", "agi"]
not_import_stats = ["icon", "name", "drop", "classReq", "lore", "tier", "dropInfo", "quest", "armourMaterial"]
maximized_stats = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"]
build_unique_stats = ["averageDps", "atkSpd"]
item_only_stats = ["id", "displayName", "restrict", "allowCraftsman", "category", "type", "majorIds"] + skill_point_types

def load_game_data(filepath):
    """
    Loads the Wynncraft items JSON, culls heavy/redundant data, 
    and sorts them into sub-dictionaries mapped by item ID.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    items_by_type = {
        "Helmet": {}, "Chestplate": {}, "Leggings": {}, "Boots": {},
        "Ring": {}, "Bracelet": {}, "Necklace": {},
        "Dagger": {}, "Spear": {}, "Bow": {}, "Wand": {}, "Relik": {}
    }

    type_mapping = {
        "helmet": "Helmet", "chestplate": "Chestplate", 
        "leggings": "Leggings", "boots": "Boots",
        "ring": "Ring", "bracelet": "Bracelet", "necklace": "Necklace",
        "dagger": "Dagger", "spear": "Spear", "bow": "Bow", 
        "wand": "Wand", "relik": "Relik"
    }

    for item in data.get("items", []):
        raw_type = item.get("type")
        if raw_type in type_mapping:
            mapped_type = type_mapping[raw_type]
            item_id = item.get("id")
            
            if item_id is not None:
                for stat_type in not_import_stats:
                    item.pop(stat_type, None)
                items_by_type[mapped_type][item_id] = item

    sets_data = data.get("sets", {})
    return items_by_type, sets_data

def stat_to_max(stat):
    """Returns max stat from range. Ex: '0-18' -> 18"""
    if isinstance(stat, str) and '-' in stat:
        return int(stat.split('-')[-1])
    return stat
