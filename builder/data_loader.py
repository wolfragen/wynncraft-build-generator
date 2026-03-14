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
                
                # Apply stat_to_max to all remaining fields that might be ranges
                for key in list(item.keys()):
                    if key not in ["displayName", "type", "category", "majorIds", "restrict", "allowCraftsman"]:
                        item[key] = stat_to_max(item[key])
                
                items_by_type[mapped_type][item_id] = item

    sets_data = data.get("sets", {})
    return items_by_type, sets_data

def stat_to_max(stat):
    """Returns max stat from range. Ex: '0-18' -> 18, '-10--5' -> -5, '-10' -> -10"""
    if stat is None:
        return 0
    if isinstance(stat, list):
        return stat[1] if len(stat) > 1 else (stat[0] if stat else 0)
    if not isinstance(stat, str):
        return stat
    
    # Try direct conversion (handles "-10", "10")
    try:
        return int(stat)
    except ValueError:
        pass
    
    # Handle ranges
    if '-' in stat:
        # Find the range separator: a hyphen preceded by a digit
        for i in range(1, len(stat)):
            if stat[i] == '-' and stat[i-1].isdigit():
                try:
                    return int(stat[i+1:])
                except ValueError:
                    pass
    return stat
