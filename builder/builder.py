import copy
import json

# --- IMPORT ITEMS ---

import json

not_import_stats = ["category", "icon", "name", "drop", "classReq", "lore", "category", "tier"]
maximized_stats = ["lvl", "strReq", "dexReq", "intReq", "defReq", "agiReq"] # stats where max matters
build_unique_stats = ["averageDps", "atkSpd"] # stats that only appear once
item_only_stats = ["id", "displayName", "restrict", "allowCraftsman", "category"] # stats like item displayName that are not used in build stat aggregation
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
skill_point_types = ["str", "dex", "int", "def", "agi"]

class BuildInfo:
    def __init__(self, weapon_type, items=None, skill_points=None, available_skill_points = None, exclusive_flags=None):
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
        
        # TODO Boolean flags for mutually exclusive items (Hive + Ornate Shadow)
        self.exclusive_flags = exclusive_flags or set()

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
        new_build = BuildInfo(items=new_items, skill_points=new_sp, exclusive_flags=new_flags)
        
        # We know skill points can be attributed at this point since can_equip was passed
        new_build.available_skill_points = new_available_skill_points
        new_build.skill_points = copy.deepcopy(self.skill_points)
        for sk in new_attributions.keys():
            new_build.skill_points[sk]["attributed"] += new_attributions[sk]

        # TODO add new_build.exclusive_flags based on the newly added item (Hive, Ornate Shadow).
        
        return new_build
    
    def calculate_stats(self):
        self.stats = {}
        for item in self.items.values:
            if item != None:
                for stat in item:
                    # skill points
                    if stat in skill_point_types:
                        if item["category"] == "weapon": # TODO Or Crafted item
                            self.skill_points[stat]["items_not_counting"] += stat_to_max(item[stat]) # crafted can have variable skill points
                        else:
                            self.skill_points[stat]["items_counting"] += item[stat]
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

def generate_builds(query="", imposed_items=None, weapon_type="Bow"):
    best_build = None
    best_score = -float('inf')

    # TODO create out of query instead
    score_item_function = score_item_spellDmg
    
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
            current_score = score_build_spellDmg(current_build)
            if current_score > best_score:
                best_score = current_score
                best_build = current_build
                print(f"New best build found! Score: {best_score}")
                print(f"Items: {(item['displayName'] for item in best_build.items)}")
            return

        # Recursive Step: Try picking a slot to fill next (Iterating orders)
        for i, slot_name in enumerate(slots_left):
            
            # Map Ring1 and Ring2 to the overarching "Ring" type for your search
            item_type = "Ring" if slot_name in ["Ring1", "Ring2"] else slot_name
            
            # Get candidates for this specific slot at this specific point in the tree
            candidates = search_items(item_type, current_build, score_item_function)
            
            for item in candidates:
                # Equip item and create the next branch of the tree
                next_build = current_build.add_item(slot_name, item)
                
                # Remove the slot we just filled from the remaining list
                next_slots_left = slots_left[:i] + slots_left[i+1:]
                
                # Dive deeper into the tree
                dfs(next_build, next_slots_left)

    # 3. Start Search
    print("Starting build generation...")
    dfs(initial_build, remaining_slots)
    return best_build





def stat_to_max(stat):
    # returns max stat from range. Ex: stat_to_max("0-18") -> int(18)
    if type(stat) == str and '-' in stat:
        stat.split('-')
        stat = int(stat[-1])
    return stat


def score_build_spellDmg(build_info : BuildInfo, query : ...):
    # TODO use proper formulas with elements and skill points, adjust fields names
    spellRaw = build_info.stats["sdRaw"] if "sdRaw" in build_info.stats else 0
    spellPct = build_info.stats["sdPct"] if "sdPct" in build_info.stats else 0
    dps = build_info.stats["averageDps"] if "averageDps" in build_info.stats else 0
    skStr = build_info.skill_points["str"]["attributed"] + build_info.skill_points["str"]["items_counting"] + build_info.skill_points["str"]["items_not_counting"]
    return (dps+spellRaw)*(1+spellPct/100)*(1+skStr*(0.7/150)) # approx

def score_item_spellDmg(item : dict, partial_build_info : BuildInfo):
    # TODO use proper formulas with elements, adjust fields names
    # Use both items stats and partial build to evaluate actual impact
    spellRawBuild = partial_build_info.stats["sdRaw"] if "sdRaw" in partial_build_info.stats else 0
    spellRawItem = stat_to_max(item["sdRaw"]) if "sdRaw" in item else 0
    spellPctBuild = partial_build_info.stats["sdPct"] if "sdPct" in partial_build_info.stats else 0
    spellPctItem = stat_to_max(item["sdPct"]) if "sdPct" in item else 0
    dps = partial_build_info.stats["averageDps"] if "averageDps" in partial_build_info.stats else 500 # 500 is not perfect but more representative of a weapon than 0
    return (dps+spellRawBuild+spellRawItem)*(1+(spellPctBuild+spellPctItem)/100)

def penalize_skill_point_reqs(item : dict, score : float, penalty_strength : float, partial_build_info : BuildInfo):
    # TODO apply a % penalty from skill points beyond current attribution 
    return 0


def can_equip(item : dict, partial_build_info : BuildInfo):

    # TODO: First do Mututally exclusive flags (Hive, Ornate Shadow)

    skill_points_available = partial_build_info.available_skill_points
    skill_points_counting = {sk:partial_build_info["attributed"][sk]+partial_build_info["items_counting"][sk]
                             for sk in skill_point_types}
    skill_points_needed = {sk:item[sk+"Req"] for sk in skill_point_types}

    # negative skill points must be compensated by attributed ones to keep currently equipped build.
    if not (item["category"] == "weapon"): # TODO Or Crafted item
        for sk in skill_point_types:
            if sk in item and item[sk] < 0:
                skill_points_needed[sk] -= item[sk] # increases needed skill accordingly

    # check if equippable directly
    valid = all(skill_points_needed[sk] <= skill_points_counting[sk] for sk in skill_point_types)
    newly_attributed = None
    # try to attribute manually
    if not valid:
        newly_attributed = {sk:max(0, skill_points_needed[sk]-skill_points_counting[sk]) for sk in skill_point_types}
        if any(partial_build_info["attributed"][sk]+newly_attributed[sk]>100 for sk in skill_point_types): # todo verify if negative sk allow for more than 100 manual
            return False, {}, 0 # Needs more than 100 manual skill points
        skill_points_available -= sum(newly_attributed.values)
        if skill_points_available < 0:
            return False, {}, 0 # Not enough skill points    

    return valid, newly_attributed, skill_points_available

def search_items(item_type : str, partial_build_info : BuildInfo, score_item_function, range_of_penalty : int = 0):
    # Search items that are equippable and maximize score
    # range of penalty is the number of supplementary items with less restrictive skill requirements to try. 0 is no penalty -> only the "best"
    # returns a list of tuple(item, new_skill_point_attributions, new_available_skill_points)
    max_scores = [(None, 0) for _ in range(range_of_penalty+1)] # (best_item, max_score) for normal max and penalty levels
    for item in items_database[item_type]:
        equipable, new_attributions, new_available_skill_points = can_equip(item, partial_build_info)
        item_info = item, new_attributions, new_available_skill_points
        if equipable:
            score = score_item_function(item_info, partial_build_info)
            if score > max_scores[0][1]: # no penalty, pure score
                max_scores[0] = (item_info, score)
            else: # this else + break avoid duplicates
                for i in range(1, range_of_penalty+1): # with increasing penalty
                    penalized_score = penalize_skill_point_reqs(new_attributions, score, i)
                    if penalized_score > max_scores[i][1]:
                        max_scores[i] = (item_info, penalized_score)
                        break
    # Since a lot of items don't have requirements we are reasonnably sure to have something. 
    # If <range_of_penalty> is way too high it could stop being enough
    return [item_info for item_info, _ in max_scores]


items_database, sets_database = load_game_data('items.json')

if __name__ == "__main__":
    generate_builds()


    