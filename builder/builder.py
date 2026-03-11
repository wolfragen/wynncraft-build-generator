import copy
import json
from numpy import clip as clamp

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

def generate_builds(imposed_items=[], weapon_type="Wand", equip_orders=None, top_k=3):
    best_build = None
    best_score = -float('inf')
    
    # 1. Setup Initial State
    initial_build = BuildInfo(weapon_type=weapon_type)
    if imposed_items:
        for slot, itemName in imposed_items:
            # get item from displayName in database:
            item = None
            for it in items_database[slot].values():
                if it["displayName"] == itemName:
                    item = it
                    break
            if item is None:
                raise ValueError(f"Imposed item {itemName} not found in database for slot {slot}.")
            equippable, newly_attributed, new_available = can_equip(item, initial_build)
            assert equippable, f"Imposed items cannot be equipped together."
            item_info = (item, newly_attributed, new_available)
            initial_build = initial_build.add_item(slot, item_info)
            
    # Default order if none is provided. You can prioritize high-req slots first (like Chestplate/Leggings)
    if not equip_orders:
        equip_orders = [
            ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", weapon_type]
        ]

    # 2. The Recursive DFS (Now strictly follows the given order)
    def dfs(current_build: BuildInfo, current_order):
        nonlocal best_build, best_score

        # Base Case: The order is complete, meaning the build is full
        if not current_order:
            current_build.calculate_stats()
            current_score = score_build_function(current_build)
            
            if current_score > best_score:
                best_score = current_score
                best_build = current_build
                print(f"New best build found! Score: {best_score}")
                print(f" - ".join(slot + ": " + item['displayName'] for slot, item in best_build.items.items() if item is not None))
                print(f"Available SP: {best_build.available_skill_points}")
                print(f"Stats: {best_build.stats}")
                print("------")
            return

        # Recursive Step: Take the FIRST slot in the current order
        slot_name = current_order[0]
        
        # If the slot is already filled (e.g., via imposed_items), skip it and move forward
        if current_build.items.get(slot_name) is not None:
            dfs(current_build, current_order[1:])
            return
            
        # Map Ring1 and Ring2 to the overarching "Ring" type for your search
        item_type = "Ring" if slot_name in ["Ring1", "Ring2"] else slot_name
        
        # Get the top K candidates for this specific slot
        candidates = search_items(item_type, current_build, score_item_function, top_k)
        
        for item_info in candidates:
            # Equip item and create the next branch of the tree
            next_build = current_build.add_item(slot_name, item_info)
            
            # Dive deeper into the tree, passing the REST of the order
            dfs(next_build, current_order[1:])

    # 3. Start Search
    print("Starting build generation...")
    for order in equip_orders:
        print(f"Evaluating order: {order}")
        dfs(initial_build, order)
        
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


def search_items(item_type: str, partial_build_info: BuildInfo, score_item_function, top_k: int):
    """
    Search items that are equippable and maximize score.
    Returns a list of up to top_k tuples:
    (item, new_skill_point_attributions, new_available_skill_points)
    """
    top_items = []  # list of tuples: (item_info, score)

    for item in items_database[item_type].values():
        equipable, new_attributions, new_available_skill_points = can_equip(item, partial_build_info)

        if not equipable:
            continue

        score = score_item_function(item, partial_build_info)
        item_info = (item, new_attributions, new_available_skill_points)

        # Insert and keep only top_k sorted by score (descending)
        top_items.append((item_info, score))
        top_items.sort(key=lambda x: x[1], reverse=True)

        if len(top_items) > top_k:
            top_items.pop()  # remove lowest

    return [item_info for item_info, _ in top_items]


# --- SCORING FUNCTIONS ---

def score_build_spellDmg(build_info : BuildInfo):
    # TODO use proper formulas with elements and skill points, adjust fields names
    spellRaw = build_info.stats["sdRaw"] if "sdRaw" in build_info.stats else 0
    spellPct = build_info.stats["sdPct"] if "sdPct" in build_info.stats else 0
    dps = build_info.stats["averageDps"] if "averageDps" in build_info.stats else 0
    skStr = build_info.skill_points["str"]["attributed"] + build_info.skill_points["str"]["items_counting"] + build_info.skill_points["str"]["items_not_counting"]
    return max(0, dps+spellRaw)*(1+spellPct/100)*(1+clamp(skStr, 0, 150)*(0.7/150)) # approx

def score_build_ehp(build_info : BuildInfo):
    # TODO use proper formulas with elements and skill points, adjust fields names
    hp = build_info.stats["hp"] if "hp" in build_info.stats else 0
    hp += build_info.stats["hpBonus"] if "hpBonus" in build_info.stats else 0
    skDef = build_info.skill_points["def"]["attributed"] + build_info.skill_points["def"]["items_counting"] + build_info.skill_points["def"]["items_not_counting"]
    skAgi = build_info.skill_points["agi"]["attributed"] + build_info.skill_points["agi"]["items_counting"] + build_info.skill_points["agi"]["items_not_counting"]
    return hp*(1+clamp(skDef, 0, 150)*(0.7/150))*(1+clamp(skAgi, 0, 150)*(0.7/150)) # approx

def score_build_raw_hp_hpr_thorns_reflex(build_info : BuildInfo):
    hpWeight = 1
    hprWeight = 8

    hp = build_info.stats["hp"] if "hp" in build_info.stats else 0
    hp += build_info.stats["hpBonus"] if "hpBonus" in build_info.stats else 0
    hprRaw = build_info.stats["hprRaw"] if "hprRaw" in build_info.stats else 0
    hrpPct = build_info.stats["hprPct"] if "hprPct" in build_info.stats else 0
    trueHpr = hprRaw*(1+hrpPct/100)
    thorns = build_info.stats["thorns"] if "thorns" in build_info.stats else 0
    reflexion = build_info.stats["ref"] if "ref" in build_info.stats else 0
    if (thorns < 100 or reflexion < 100):
        return 0
    else:
        return hp*hpWeight + trueHpr*hprWeight

def score_item_spellDmg(item : dict, partial_build_info : BuildInfo):
    # TODO use proper formulas with elements, adjust fields names
    # Use both items stats and partial build to evaluate actual impact
    spellRawBuild = partial_build_info.stats["sdRaw"] if "sdRaw" in partial_build_info.stats else 0
    spellRawItem = stat_to_max(item["sdRaw"]) if "sdRaw" in item else 0
    spellPctBuild = partial_build_info.stats["sdPct"] if "sdPct" in partial_build_info.stats else 0
    spellPctItem = stat_to_max(item["sdPct"]) if "sdPct" in item else 0
    skStrBuild = partial_build_info.skill_points["str"]["attributed"] + partial_build_info.skill_points["str"]["items_counting"] + partial_build_info.skill_points["str"]["items_not_counting"]
    skStrItem = stat_to_max(item["str"]) if "str" in item else 0
    if item.get("category") == "weapon":
        dps = item["averageDps"] if "averageDps" in item else 0
    else:
        dps = partial_build_info.stats["averageDps"] if "averageDps" in partial_build_info.stats else 0 # 160 is not perfect but more representative of a weapon than 0
    return max(0, dps+spellRawBuild+spellRawItem)*(1+(spellPctBuild+spellPctItem)/100)*(1+clamp(skStrBuild+skStrItem, 0, 150)*(0.7/150)) # approx

def score_item_ehp(item : dict, partial_build_info : BuildInfo):
    # TODO use proper formulas with elements, adjust fields names
    hpBuild = partial_build_info.stats["hp"] if "hp" in partial_build_info.stats else 0
    hpBuild += partial_build_info.stats["hpBonus"] if "hpBonus" in partial_build_info.stats else 0
    hpItem = stat_to_max(item["hp"]) if "hp" in item else 0
    hpItem += stat_to_max(item["hpBonus"]) if "hpBonus" in item else 0
    skDefBuild = partial_build_info.skill_points["def"]["attributed"] + partial_build_info.skill_points["def"]["items_counting"] + partial_build_info.skill_points["def"]["items_not_counting"]
    skDefItem = stat_to_max(item["def"]) if "def" in item else 0
    skAgiBuild = partial_build_info.skill_points["agi"]["attributed"] + partial_build_info.skill_points["agi"]["items_counting"] + partial_build_info.skill_points["agi"]["items_not_counting"]
    skAgiItem = stat_to_max(item["agi"]) if "agi" in item else 0
    
    return (hpBuild+hpItem)*(1+clamp(skDefBuild+skDefItem, 0, 150)*(0.7/150))*(1+clamp(skAgiBuild+skAgiItem, 0, 150)*(0.7/150)) # approx

def score_item_raw_hp_hpr_thorns_reflex(item : dict, partial_build_info : BuildInfo):
    hpWeight = 2
    hprWeight = 16
    thornsWeight = 8
    reflexionWeight = 8


    hpBuild = partial_build_info.stats["hp"] if "hp" in partial_build_info.stats else 0
    hpBuild += partial_build_info.stats["hpBonus"] if "hpBonus" in partial_build_info.stats else 0
    hpItem = stat_to_max(item["hp"]) if "hp" in item else 0
    hpItem += stat_to_max(item["hpBonus"]) if "hpBonus" in item else 0
    hprRawBuild = partial_build_info.stats["hprRaw"] if "hprRaw" in partial_build_info.stats else 0
    hprRawItem = stat_to_max(item["hprRaw"]) if "hprRaw" in item else 0
    hrpPctBuild = partial_build_info.stats["hprPct"] if "hprPct" in partial_build_info.stats else 0
    hrpPctItem = stat_to_max(item["hprPct"]) if "hprPct" in item else 0
    trueHpr = (hprRawBuild+hprRawItem)*(1+(hrpPctBuild+hrpPctItem)/100)

    thornsBuild = partial_build_info.stats["thorns"] if "thorns" in partial_build_info.stats else 0
    thornsItem = stat_to_max(item["thorns"]) if "thorns" in item else 0
    reflexionBuild = partial_build_info.stats["ref"] if "ref" in partial_build_info.stats else 0
    reflexionItem = stat_to_max(item["ref"]) if "ref" in item else 0
    
    if thornsBuild+thornsItem > 100:
        thornsWeight = 0
    if reflexionBuild+reflexionItem > 100:
        reflexionWeight = 0
    
    return (hpBuild+hpItem)*hpWeight + trueHpr*hprWeight + (thornsBuild+thornsItem)*thornsWeight + (reflexionBuild+reflexionItem)*reflexionWeight


def score_build_custom(build_info : BuildInfo):
    return 0 # replace with your objective

def score_item_custom(item : dict, partial_build_info : BuildInfo):
    return 0 # replace with your objective

# --- MAIN EXECUTION ---



items_database, sets_database = load_game_data('items.json')

if __name__ == "__main__":
    orders_to_test = [
        # Standard heavy-to-light armor, then accessories, then weapon
        ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Wand"],
        
        # Sometimes weapon first helps define the required elements early
        ["Wand", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Chestplate"]
    ]
    
    # Try increasing top_k to 4 or 5 now that permutations are under control!
    '''
    score_build_function  = score_build_spellDmg
    score_item_function = score_item_spellDmg
    generate_builds(imposed_items=[("Wand", "Quetzalcoatl")], weapon_type="Wand", equip_orders=orders_to_test, top_k=3)
    '''
    score_build_function  = score_build_raw_hp_hpr_thorns_reflex
    score_item_function = score_item_raw_hp_hpr_thorns_reflex
    #generate_builds(imposed_items=[("Wand", "Depressing Stick"), ("Chestplate", "About-Face")], weapon_type="Wand", equip_orders=orders_to_test, top_k=6)
    generate_builds(imposed_items=[("Wand", "Depressing Stick")], weapon_type="Wand", equip_orders=orders_to_test, top_k=4)
'''
TODO make incremental improvements
TODO make progress indicators
TODO add crafted
'''