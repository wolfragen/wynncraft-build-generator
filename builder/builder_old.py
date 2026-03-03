import copy
import json

# --- IMPORT ITEMS ---

import json

skill_point_types = ["str", "dex", "int", "def", "agi"]
not_import_stats = ["icon", "name", "drop", "classReq", "lore", "tier", "dropInfo", "quest", "armourMaterial"]
maximized_stats = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"] # stats where max matters
build_unique_stats = ["averageDps", "atkSpd"] # stats that only appear once
item_only_stats = ["id", "displayName", "restrict", "allowCraftsman", "category", "type"] # stats like item displayName that are not used in build stat aggregation
item_only_stats += skill_point_types # fix counting skill points twice

#added_stats = ["nDam", "sdPct", "sdRaw", ...] # stats that can be += like spellDmg - TODO do it properly. TEMP : not in the others

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


# --- BASE DATA STRUCTURES ---

class BuildInfo:
    def __init__(self, weapon_type, items=None, skill_points=None, available_skill_points = None, max_reqs=None, exclusive_flags=None):
        self.weapon_type = weapon_type
        # The 9 slots of a build
        self.items = items or {
            weapon_type: None, "Helmet": None, "Chestplate": None, 
            "Leggings": None, "Boots": None, "Ring1": None, 
            "Ring2": None, "Bracelet": None, "Necklace": None
        }

        # 5 types, 3 states (User to implement exact structure)
        # TODO replace dict by lists when it starts working
        self.available_skill_points = available_skill_points or 200
        self.skill_points = skill_points or {sk:{"attributed":0, "items_counting":0, "items_not_counting":0} 
                                             for sk in skill_point_types}
        self.max_reqs = max_reqs or {sk: -150 for sk in skill_point_types}

        # TODO Boolean flags for mutually exclusive items (Hive + Ornate Shadow)
        self.exclusive_flags = exclusive_flags or set()

        self.stats = {}

    def add_item(self, slot_name, item_info):
        """
        Returns a new BuildInfo with the item equipped.
        This ensures we don't corrupt the state of parent nodes in the DFS tree.
        """
        
        item, new_attributions, new_available_skill_points = item_info

        new_items = self.items.copy()
        new_items[slot_name] = item
        
        # Deepcopy because skill points structure has nested dictionaries/states
        new_sp = copy.deepcopy(self.skill_points)
        new_flags = self.exclusive_flags.copy()
        
        # Create the new state
        new_build = BuildInfo(self.weapon_type, items=new_items, skill_points=new_sp, exclusive_flags=new_flags)
        
        # Update max_reqs based on the newly added item
        new_build.max_reqs = self.max_reqs.copy()
        for sk in skill_point_types:
            skReq = sk + "Req"
            if skReq in item and item[skReq] > new_build.max_reqs[sk]:
                new_build.max_reqs[sk] = item[skReq]

        # We know skill points can be attributed at this point since can_equip was passed
        new_build.available_skill_points = new_available_skill_points
        for sk in new_attributions.keys():
            new_build.skill_points[sk]["attributed"] += new_attributions[sk]

        for sk in skill_point_types:
            if sk in item:
                if item["category"] == "weapon": # TODO Or Crafted item
                    new_build.skill_points[sk]["items_not_counting"] += stat_to_max(item[sk]) # crafted can have variable skill points
                else:
                    new_build.skill_points[sk]["items_counting"] += item[sk]

        # TODO add new_build.exclusive_flags based on the newly added item (Hive, Ornate Shadow).
        
        return new_build
    
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
        


# --- THE SEARCH ALGORITHM ---

def generate_builds(imposed_items=None, weapon_type="Bow"):
    best_build = None
    best_score = -float('inf')
    
    # 1. Setup Initial State
    initial_build = BuildInfo(weapon_type=weapon_type)
    if imposed_items:
        for slot, item in imposed_items.items():
            equippable, item_info = can_equip(item, initial_build)
            assert equippable, f"Imposed items cannot be equipped together."
            initial_build = initial_build.add_item(slot, item_info)
            
    all_slots = initial_build.items.keys()
    
    # Only search slots that haven't been imposed
    remaining_slots = [s for s in all_slots if initial_build.items[s] is None]

    # 2. The Recursive DFS
    def dfs(current_build : BuildInfo, slots_left):
        nonlocal best_build, best_score

        # Base Case: The build is complete
        if not slots_left:
            current_build.calculate_stats()
            current_score = score_build_function(current_build)
            #print(f"Completed a build with score {current_score}")
            if current_score > best_score:
                best_score = current_score
                best_build = current_build
                print(f"New best build found! Score: {best_score}")
                print(f" - ".join(slot + ": " + item['displayName'] for slot, item in best_build.items.items() if item is not None))
                #print(f"Skill Points: {best_build.skill_points}")
                print(f"Available SP: {best_build.available_skill_points}")
                print(f"Stats: {best_build.stats}")
                print("------")
            return

        # Recursive Step: Try picking a slot to fill next (Iterating orders)
        for i, slot_name in enumerate(slots_left):
            
            # Map Ring1 and Ring2 to the overarching "Ring" type for your search
            item_type = "Ring" if slot_name in ["Ring1", "Ring2"] else slot_name
            
            # Get candidates for this specific slot at this specific point in the tree
            candidates = search_items(item_type, current_build, score_item_function)
            
            for item_info in candidates:
                #print(f"Trying to equip {item_info[0]['displayName']} in slot {slot_name} with new attributions {item_info[1]} (available: {item_info[2]})")
                # Equip item and create the next branch of the tree
                next_build = current_build.add_item(slot_name, item_info)
                
                # Remove the slot we just filled from the remaining list
                next_slots_left = slots_left[:i] + slots_left[i+1:]
                
                # Dive deeper into the tree
                dfs(next_build, next_slots_left)

    # 3. Start Search
    print("Starting build generation...")
    dfs(initial_build, remaining_slots)
    return best_build



def stat_to_max(stat):
    # Returns max stat from range. Ex: "0-18" -> 18
    if isinstance(stat, str) and '-' in stat:
        return int(stat.split('-')[-1])
    return stat




def penalize_skill_point_reqs(item : dict, score : float, penalty_strength : float, partial_build_info : BuildInfo):
    # TODO apply a % penalty from skill points beyond current attribution 
    return 0


def can_equip(item: dict, partial_build_info: BuildInfo):
    
    # TODO: Mutually exclusive flags (Hive, Ornate Shadow)

    newly_attributed = {}
    
    for sk in skill_point_types:
        current_attr = partial_build_info.skill_points[sk]["attributed"]
        current_items = partial_build_info.skill_points[sk]["items_counting"]
        item_req = item.get(sk + "Req", -150)
        
        # Weapons don't provide skill points to armor in Wynncraft
        if item.get("category") == "weapon":
            item_bonus = 0
        else:
            item_bonus = stat_to_max(item.get(sk, 0))
            
        # STEP 1: Do we have enough points just to hold the item?
        needed_to_equip = max(0, item_req - (current_attr + current_items))
        
        # STEP 2: After putting it on, do we drop below the reqs of older items?
        net_after_equip = current_attr + current_items + needed_to_equip + item_bonus
        needed_to_sustain = max(0, partial_build_info.max_reqs[sk] - net_after_equip)
        
        # Total points we must spend this turn
        newly_attributed[sk] = needed_to_equip + needed_to_sustain
        
        # Fail if this pushes us over the 100 manual point cap
        if current_attr + newly_attributed[sk] > 100:
            return False, {}, 0
            
    # Check if we have enough total available points left (from the 200 pool)
    total_needed = sum(newly_attributed.values())
    if partial_build_info.available_skill_points < total_needed:
        return False, {}, 0
        
    new_available = partial_build_info.available_skill_points - total_needed
    return True, newly_attributed, new_available


def search_items(item_type: str, partial_build_info: BuildInfo, score_item_function):
    """
    Search items that are equippable and maximize score.
    Returns a list of up to 3 tuples:
    (item, new_skill_point_attributions, new_available_skill_points)
    """

    top_items = []  # list of tuples: (item_info, score)

    for item in items_database[item_type].values():
        equipable, new_attributions, new_available_skill_points = can_equip(item, partial_build_info)

        if not equipable:
            continue

        score = score_item_function(item, partial_build_info)
        item_info = (item, new_attributions, new_available_skill_points)

        # Insert and keep only top 3 sorted by score (descending)
        top_items.append((item_info, score))
        top_items.sort(key=lambda x: x[1], reverse=True)

        if len(top_items) > 3:
            top_items.pop()  # remove lowest

    return [item_info for item_info, _ in top_items]


def score_build_spellDmg(build_info : BuildInfo):
    # TODO use proper formulas with elements and skill points, adjust fields names
    spellRaw = build_info.stats["sdRaw"] if "sdRaw" in build_info.stats else 0
    spellPct = build_info.stats["sdPct"] if "sdPct" in build_info.stats else 0
    dps = build_info.stats["averageDps"] if "averageDps" in build_info.stats else 0
    skStr = build_info.skill_points["str"]["attributed"] + build_info.skill_points["str"]["items_counting"] + build_info.skill_points["str"]["items_not_counting"]
    return max(0, dps+spellRaw)*(1+spellPct/100)*(1+max(0,skStr)*(0.7/150)) # approx

def score_item_spellDmg(item : dict, partial_build_info : BuildInfo):
    # TODO use proper formulas with elements, adjust fields names
    # Use both items stats and partial build to evaluate actual impact
    spellRawBuild = partial_build_info.stats["sdRaw"] if "sdRaw" in partial_build_info.stats else 0
    spellRawItem = stat_to_max(item["sdRaw"]) if "sdRaw" in item else 0
    spellPctBuild = partial_build_info.stats["sdPct"] if "sdPct" in partial_build_info.stats else 0
    spellPctItem = stat_to_max(item["sdPct"]) if "sdPct" in item else 0
    dps = partial_build_info.stats["averageDps"] if "averageDps" in partial_build_info.stats else 0 # 160 is not perfect but more representative of a weapon than 0
    return max(0, dps+spellRawBuild+spellRawItem)*(1+(spellPctBuild+spellPctItem)/100)

def score_item_custom(item : dict, partial_build_info : BuildInfo):
    return 0 # replace with your objective

def score_build_custom(build_info : BuildInfo):
    return 0 # replace with your objective

score_build_function  = score_build_spellDmg
score_item_function = score_item_spellDmg

items_database, sets_database = load_game_data('items.json')

if __name__ == "__main__":
    generate_builds(imposed_items=None, weapon_type="Wand")


'''
TODO do skill req pernalty to branch into more possibilities
TODO make incremental improvements
TODO make progress indicators
TODO add crafted
'''