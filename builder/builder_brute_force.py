import copy
import json
import time
from numpy import clip as clamp

skill_point_types = ["str", "dex", "int", "def", "agi"]
not_import_stats = ["icon", "name", "drop", "classReq", "lore", "tier", "dropInfo", "quest", "armourMaterial"]
maximized_stats = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"] # stats where max matters
build_unique_stats = ["averageDps", "atkSpd"] # stats that only appear once
# Updated exclusion list to prevent double-counting
item_only_stats = ["id", "displayName", "restrict", "allowCraftsman", "category", "type", 
                   "str", "dex", "int", "def", "agi", "strReq", "dexReq", "intReq", "defReq", "agiReq"]

def load_game_data(filepath):
    """
    Loads the Wynncraft items JSON, culls heavy/redundant data, 
    and sorts them into sub-dictionaries mapped by item ID.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize the dictionary with specific types as keys
    items_by_type = {
        "Helmet": {}, "Chestplate": {}, "Leggings": {}, "Boots": {},
        "Ring": {}, "Bracelet": {}, "Necklace": {},
        "Dagger": {}, "Spear": {}, "Bow": {}, "Wand": {}, "Relik": {}
    }

    # Map the JSON 'type' values to our exact dictionary keys
    type_mapping = {
        "helmet": "Helmet", "chestplate": "Chestplate", 
        "leggings": "Leggings", "boots": "Boots",
        "ring": "Ring", "bracelet": "Bracelet", "necklace": "Necklace",
        "dagger": "Dagger", "spear": "Spear", "bow": "Bow", 
        "wand": "Wand", "relik": "Relik"
    }

    for item in data.get("items", []):
        raw_type = item.get("type")
        
        # Only process items that match our 12 mapped types
        if raw_type in type_mapping:
            mapped_type = type_mapping[raw_type]
            item_id = item.get("id")
            
            if item_id is not None:
                # Cull useless fields
                for stat_type in not_import_stats:
                    item.pop(stat_type, None)
                

                # Store the item in the sub-dictionary using its ID as the key
                items_by_type[mapped_type][item_id] = item

    sets_data = data.get("sets", {})

    return items_by_type, sets_data

def cull_dominated_items(items_by_type):
    """
    Removes items that are strictly worse than another item in the same category.
    An item is culled if another item has:
    - Higher or equal 'good' stats (damage, skill bonuses, sdRaw, etc.)
    - Lower or equal 'bad' stats (lvl, strReq, dexReq, etc.)
    - AND at least one stat is strictly better.
    """
    culled_db = {}
    
    # Define which stats we want to compare
    # 'Bad' stats: Lower is better
    requirements = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"]
    
    for item_type, items_dict in items_by_type.items():
        original_list = list(items_dict.values())
        to_remove = set()
        
        # We only compare stats that actually exist in this item category
        # to avoid comparing "spell damage" on a warrior spear vs a helmet.
        all_relevant_keys = set()
        for item in original_list:
            for key in item.keys():
                if key not in ["id", "displayName", "type", "category", "restrict", "allowCraftsman"]:
                    all_relevant_keys.add(key)
        
        for i in range(len(original_list)):
            item_a = original_list[i]
            for j in range(len(original_list)):
                if i == j: continue
                item_b = original_list[j]
                
                # Check if B dominates A (Is B better than A in every way?)
                b_is_better_or_equal = True
                at_least_one_strictly_better = False
                
                for key in all_relevant_keys:
                    val_a = stat_to_max(item_a.get(key, 0))
                    val_b = stat_to_max(item_b.get(key, 0))
                    
                    if key in requirements:
                        # For requirements, lower is better. 
                        # B is worse if its req is higher than A's.
                        if val_b > val_a:
                            b_is_better_or_equal = False
                            break
                        if val_b < val_a:
                            at_least_one_strictly_better = True
                    else:
                        # For all other stats, higher is better.
                        # B is worse if its stat is lower than A's.
                        if val_b < val_a:
                            b_is_better_or_equal = False
                            break
                        if val_b > val_a:
                            at_least_one_strictly_better = True
                
                if b_is_better_or_equal and at_least_one_strictly_better:
                    to_remove.add(item_a['id'])
                    break # A is dominated, no need to check other items
        
        # Rebuild the dictionary excluding the dominated items
        culled_db[item_type] = {
            id: item for id, item in items_dict.items() if id not in to_remove
        }
        
        print(f"{item_type}: Culled {len(to_remove)} dominated items. ({len(culled_db[item_type])} remaining)")
        
    return culled_db

def cull_by_stats(items_by_type, important_stats):
    """
    Aggressively culls items based on a specific subset of stats.
    An item is removed if another item provides better or equal benefits 
    for the same or lower requirements.
    """
    requirements = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"]
    culled_db = {}

    for item_type, items_dict in items_by_type.items():
        original_list = list(items_dict.values())
        to_remove = set()
        
        # O(N^2) comparison within each item type
        for i in range(len(original_list)):
            item_a = original_list[i]
            for j in range(len(original_list)):
                if i == j: continue
                item_b = original_list[j]
                
                # We want to see if B is 'better' than A
                b_is_better_or_equal = True
                at_least_one_strictly_better = False
                
                # 1. Compare Benefits (Higher is better)
                for stat in important_stats:
                    val_a = stat_to_max(item_a.get(stat, 0))
                    val_b = stat_to_max(item_b.get(stat, 0))
                    
                    if val_b < val_a:
                        b_is_better_or_equal = False
                        break
                    if val_b > val_a:
                        at_least_one_strictly_better = True
                
                if not b_is_better_or_equal: continue

                # 2. Compare Requirements (Lower is better)
                for req in requirements:
                    val_a = item_a.get(req, 0)
                    val_b = item_b.get(req, 0)
                    
                    if val_b > val_a:
                        b_is_better_or_equal = False
                        break
                    if val_b < val_a:
                        at_least_one_strictly_better = True
                
                # 3. Handle Identical Items
                # If they are exactly the same in all relevant stats, 
                # keep the one with the lower ID to avoid duplicates.
                if b_is_better_or_equal and not at_least_one_strictly_better:
                    if item_b['id'] < item_a['id']:
                        at_least_one_strictly_better = True

                if b_is_better_or_equal and at_least_one_strictly_better:
                    to_remove.add(item_a['id'])
                    break
        
        culled_db[item_type] = {
            id: item for id, item in items_dict.items() if id not in to_remove
        }
        print(f"{item_type}: Aggressively culled {len(to_remove)} items. ({len(culled_db[item_type])} remaining)")
        
    return culled_db


# Helper function (ensure this is in your script)
def stat_to_max(stat):
    if isinstance(stat, str) and '-' in stat:
        return int(stat.split('-')[-1])
    try:
        return float(stat)
    except (ValueError, TypeError):
        return 0


# --- CONFIGURATION & MATH ---


'''
def stat_to_max(stat):
    if isinstance(stat, str) and '-' in stat:
        return int(stat.split('-')[-1])
    return stat'''


# --- CORE DATA STRUCTURE ---

class BuildInfo:
    def __init__(self, weapon_type, items=None, skill_points=None, available_skill_points=200, max_reqs=None):
        self.weapon_type = weapon_type
        self.items = items or {
            "Helmet": None, "Chestplate": None, "Leggings": None, "Boots": None,
            "Ring1": None, "Ring2": None, "Bracelet": None, "Necklace": None, weapon_type: None
        }
        self.available_skill_points = available_skill_points
        self.skill_points = skill_points or {sk: {"attributed": 0, "items_counting": 0, "items_not_counting": 0} for sk in skill_point_types}
        self.max_reqs = max_reqs or {sk: 0 for sk in skill_point_types}
        self.stats = {}

    def add_item(self, slot_name, item_info):
        item, new_attr, new_avail = item_info
        
        # Manual copy is significantly faster than copy.deepcopy for brute force
        new_sp = {sk: val.copy() for sk, val in self.skill_points.items()}
        new_items = self.items.copy()
        new_items[slot_name] = item
        new_max_reqs = self.max_reqs.copy()

        for sk in skill_point_types:
            new_sp[sk]["attributed"] += new_attr.get(sk, 0)
            # Update max requirements tracked for the build
            req_val = item.get(sk + "Req", 0)
            if req_val > new_max_reqs[sk]:
                new_max_reqs[sk] = req_val
            
            # Skill point bonuses from item
            if sk in item:
                if item.get("category") == "weapon":
                    new_sp[sk]["items_not_counting"] += stat_to_max(item[sk])
                else:
                    new_sp[sk]["items_counting"] += stat_to_max(item[sk])

        return BuildInfo(self.weapon_type, new_items, new_sp, new_avail, new_max_reqs)

    def calculate_stats(self):
        self.stats = {sk:self.skill_points[sk]["attributed"]+self.skill_points[sk]["items_counting"]+self.skill_points[sk]["items_not_counting"] for sk in skill_point_types}
        self.stats["sdRaw"] = 0; self.stats["sdPct"] = 0; 
        for item in self.items.values():
            if item is not None:
                for stat in item:
                    # maximize stat case
                    if stat in maximized_stats: # stats like required lvl and skill Reqs for which only max matters
                        if not stat in self.stats:
                            self.stats[stat] = stat_to_max(item[stat])
                        else:
                            self.stats[stat] = max(self.stats[stat], stat_to_max(item[stat]))
                    elif stat in build_unique_stats: # stats like atkSpeed and averageDps that we only encounter once
                        self.stats[stat] = item[stat]
                    elif stat not in item_only_stats: # usual additive stats like sdPct or mr
                        if not stat in self.stats:
                            self.stats[stat] = stat_to_max(item[stat])
                        else:
                            self.stats[stat] += stat_to_max(item[stat])

# --- SCORING ---

def score_build_spellDmg(build_info : BuildInfo):
    # TODO use proper formulas with elements and skill points, adjust fields names
    spellRaw = build_info.stats["sdRaw"] if "sdRaw" in build_info.stats else 0
    spellPct = build_info.stats["sdPct"] if "sdPct" in build_info.stats else 0
    dps = build_info.stats["averageDps"] if "averageDps" in build_info.stats else 0
    skStr = build_info.skill_points["str"]["attributed"] + build_info.skill_points["str"]["items_counting"] + build_info.skill_points["str"]["items_not_counting"]
    return max(0, dps+spellRaw)*(1+spellPct/100)*(1+clamp(skStr, 0, 150)*(0.7/150)) # approx

# --- BRUTE FORCE ALGORITHM ---

def can_equip(item, build: BuildInfo):
    newly_attributed = {}
    for sk in skill_point_types:
        current_total = build.skill_points[sk]["attributed"] + build.skill_points[sk]["items_counting"]
        
        # Requirement for this specific item vs the highest requirement in the build so far
        goal = max(item.get(sk + "Req", 0), build.max_reqs[sk])
        
        # If item is weapon, it doesn't help satisfy armor requirements
        item_bonus = 0 if item.get("category") == "weapon" else stat_to_max(item.get(sk, 0))
        
        needed = max(0, goal - current_total)
        # Check if equipping this item breaks requirements of previous items
        if (current_total + needed + item_bonus) < goal:
            needed += max(0, goal - (current_total + needed + item_bonus))

        if build.skill_points[sk]["attributed"] + needed > 100:
            return False, None, 0
        newly_attributed[sk] = needed
        
    total_spent = sum(newly_attributed.values())
    if build.available_skill_points < total_spent:
        return False, None, 0
        
    return True, newly_attributed, build.available_skill_points - total_spent

def generate_builds_brute(items_db, weapon_type="Wand"):
    # Fixed equipment order is MANDATORY for brute force to be efficient
    order = ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", weapon_type]
    
    best_build = None
    best_score = -1
    
    # Progress Tracking
    stats = {"count": 0, "start": time.time(), "last_print": time.time()}

    def dfs(current_build, remaining_order):
        nonlocal best_build, best_score
        
        stats["count"] += 1
        now = time.time()
        if now - stats["last_print"] > 2.0:
            elapsed = now - stats["start"]
            print(f"Checked {stats['count']:,} combinations... ({stats['count']/elapsed:,.0f}/s) | Best: {best_score:.1f}")
            stats["last_print"] = now

        if not remaining_order:
            current_build.calculate_stats()
            score = score_build_spellDmg(current_build)
            if score > best_score:
                best_score = score
                best_build = current_build
                print(f"\n[NEW BEST] Score: {best_score:.2f}")
                print(" - ".join([f"{k}: {v['displayName']}" for k,v in best_build.items.items()]))
                print(f"Stats: {best_build.stats}\n")
            return

        slot = remaining_order[0]
        item_type = "Ring" if slot.startswith("Ring") else slot
        
        # BRUTE FORCE: Iterate every item in the filtered database
        for item in items_db[item_type].values():
            
                
            equippable, attr, avail = can_equip(item, current_build)
            if equippable:
                dfs(current_build.add_item(slot, (item, attr, avail)), remaining_order[1:])

    print(f"Starting Brute Force on {weapon_type} build...")
    dfs(BuildInfo(weapon_type), order)
    return best_build




if __name__ == "__main__":
    items_database, _ = load_game_data('items.json')
    # Pre-filter your items_database here if needed
    items_database = cull_dominated_items(items_database)
    print("Finished culling dominated items. Starting aggressive cull for Spell Damage...")
    
    # 2. Now, do the aggressive cull for Spell Damage
    # We include int and str because they are multipliers in your formula
    spell_dmg_stats = ["sdRaw", "sdPct"] 
    items_database = cull_by_stats(items_database, spell_dmg_stats)
    print("Finished aggressive cull. Starting brute force search for best Spell Damage build...")

    final_build = generate_builds_brute(items_database, "Wand")