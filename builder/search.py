from models import BuildInfo
from data_loader import skill_point_types, stat_to_max

def can_equip(item: dict, partial_build_info: BuildInfo):
    newly_attributed = {}
    
    for sk in skill_point_types:
        current_attr = partial_build_info.skill_points[sk]["attributed"]
        current_items = partial_build_info.skill_points[sk]["items_counting"]
        item_req = item.get(sk + "Req", -150)
        
        if item.get("category") == "weapon":
            item_bonus = 0
        else:
            item_bonus = stat_to_max(item.get(sk, 0))
            
        needed_to_equip = max(0, item_req - (current_attr + current_items))
        net_after_equip = current_attr + current_items + needed_to_equip + item_bonus
        needed_to_sustain = max(0, partial_build_info.max_reqs[sk] - net_after_equip)
        
        newly_attributed[sk] = needed_to_equip + needed_to_sustain
        
        if current_attr + newly_attributed[sk] > 100:
            return False, {}, 0
            
    total_needed = sum(newly_attributed.values())
    if partial_build_info.available_skill_points < total_needed:
        return False, {}, 0
        
    new_available = partial_build_info.available_skill_points - total_needed
    return True, newly_attributed, new_available

def search_items(item_type: str, partial_build_info: BuildInfo, score_item_function, top_k: int, items_database: dict):
    top_items = []

    for item in items_database[item_type].values():
        equipable, new_attributions, new_available_skill_points = can_equip(item, partial_build_info)

        if not equipable:
            continue

        score = score_item_function(item, partial_build_info)
        item_info = (item, new_attributions, new_available_skill_points)

        top_items.append((item_info, score))
        top_items.sort(key=lambda x: x[1], reverse=True)

        if len(top_items) > top_k:
            top_items.pop()

    return [item_info for item_info, _ in top_items]

def generate_builds(items_database, score_build_function, score_item_function, imposed_items=None, weapon_type="Wand", equip_orders=None, top_k=3):
    if imposed_items is None:
        imposed_items = []
        
    best_build = None
    best_score = -float('inf')
    
    initial_build = BuildInfo(weapon_type=weapon_type)
    if imposed_items:
        for slot, itemName in imposed_items:
            item_type = "Ring" if slot in ["Ring1", "Ring2"] else slot
            item = None
            for it in items_database[item_type].values():
                if it["displayName"] == itemName:
                    item = it
                    break
            if item is None:
                raise ValueError(f"Imposed item {itemName} not found in database for slot {slot}.")
            equippable, newly_attributed, new_available = can_equip(item, initial_build)
            assert equippable, f"Imposed items cannot be equipped together."
            item_info = (item, newly_attributed, new_available)
            initial_build = initial_build.add_item(slot, item_info)
            
    if not equip_orders:
        equip_orders = [
            ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", weapon_type]
        ]

    def dfs(current_build: BuildInfo, current_order):
        nonlocal best_build, best_score

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

        slot_name = current_order[0]
        
        if current_build.items.get(slot_name) is not None:
            dfs(current_build, current_order[1:])
            return
            
        item_type = "Ring" if slot_name in ["Ring1", "Ring2"] else slot_name
        
        candidates = search_items(item_type, current_build, score_item_function, top_k, items_database)
        
        for item_info in candidates:
            next_build = current_build.add_item(slot_name, item_info)
            dfs(next_build, current_order[1:])

    print("Starting build generation...")
    for order in equip_orders:
        print(f"Evaluating order: {order}")
        dfs(initial_build, order)
        
    return best_build

def replace_item_in_build(build: BuildInfo, slot_to_replace: str, items_database: dict, score_build_function, score_item_function, top_k=3):
    """
    Takes an existing build, removes the item in `slot_to_replace`, and finds the best replacement.
    """
    print(f"Replacing {slot_to_replace} in current build...")
    
    imposed_items = []
    for slot, item in build.items.items():
        if item is not None and slot != slot_to_replace:
            item_type = "Ring" if slot in ["Ring1", "Ring2"] else slot
            imposed_items.append((slot, item["displayName"]))
            
    return generate_builds(
        items_database=items_database,
        score_build_function=score_build_function,
        score_item_function=score_item_function,
        imposed_items=imposed_items,
        weapon_type=build.weapon_type,
        equip_orders=[[slot_to_replace]],
        top_k=top_k
    )
