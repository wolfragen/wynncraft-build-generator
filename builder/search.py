import copy
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

def item_sp_net(item):
    return sum(stat_to_max(item.get(sk, 0)) for sk in skill_point_types)

def build_base_from_items(weapon_type, items_dict):
    build = BuildInfo(weapon_type=weapon_type)
    remaining = [(slot, item) for slot, item in items_dict.items() if item is not None]
    
    # Sort remaining items: those that give the most total skill points first
    remaining.sort(key=lambda x: item_sp_net(x[1]), reverse=True)
    
    while remaining:
        progress = False
        for i in range(len(remaining)):
            slot, item = remaining[i]
            equipable, attributions, new_avail = can_equip(item, build)
            if equipable:
                build = build.add_item(slot, (item, attributions, new_avail))
                remaining.pop(i)
                progress = True
                break
        if not progress:
            return None
    return build

class BuildPool:
    def __init__(self, max_size):
        self.max_size = max_size
        self.builds = [] # list of (score, build, hash)
        self.seen_hashes = set()
        self.max_score = -float('inf')
        
    def add(self, build, score):
        b_hash = tuple(sorted((k, v["displayName"]) for k, v in build.items.items() if v))
        if b_hash in self.seen_hashes:
            return
        
        if score > self.max_score:
            self.max_score = score
            print(f"New max score: {score:.2f}")
            
        self.builds.append((score, build, b_hash))
        self.builds.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.builds) > self.max_size:
            removed = self.builds.pop()
            if (removed[2] in self.seen_hashes):
                self.seen_hashes.remove(removed[2])
            
        self.seen_hashes.add(b_hash)
        
    def get_builds(self):
        return [(b[0], b[1]) for b in self.builds]

class UnifiedBuilder:
    def __init__(self, items_database, score_build_fn, score_item_fn):
        self.items_database = items_database
        self.score_build = score_build_fn
        self.score_item = score_item_fn

    def _dfs(self, current_build, remaining_order, top_k, pool):
        if not remaining_order:
            current_build.calculate_stats()
            score = self.score_build(current_build)
            pool.add(current_build, score)
            return

        slot_name = remaining_order[0]
        
        if current_build.items.get(slot_name) is not None:
            self._dfs(current_build, remaining_order[1:], top_k, pool)
            return
            
        item_type = "Ring" if slot_name in ["Ring1", "Ring2"] else slot_name
        candidates = search_items(item_type, current_build, self.score_item, top_k, self.items_database)
        
        for item_info in candidates:
            next_build = current_build.add_item(slot_name, item_info)
            self._dfs(next_build, remaining_order[1:], top_k, pool)

    def generate(self, weapon_type, imposed_items, fill_orders, top_k, top_i):
        pool = BuildPool(top_i)
        
        base_items = {}
        for slot, item_name in imposed_items:
            item_type = "Ring" if slot in ["Ring1", "Ring2"] else slot
            item = None
            for it in self.items_database[item_type].values():
                if it["displayName"] == item_name:
                    item = it
                    break
            if item is None:
                raise ValueError(f"Imposed item {item_name} not found.")
            base_items[slot] = item
            
        base_build = build_base_from_items(weapon_type, base_items)
        if base_build is None:
            print("Could not equip the imposed items together.")
            return []
            
        for order in fill_orders:
            self._dfs(base_build, order, top_k, pool)
            
        return pool.get_builds()

    def refine(self, builds, replace_orders, imposed_slots, top_j, top_l):
        pool = BuildPool(top_l)
        
        for score, build in builds:
            # Keep the original build in the pool as a baseline
            pool.add(build, score)
            
            for order in replace_orders:
                # Filter out imposed slots from the replacement order
                actual_order = [s for s in order if s not in imposed_slots]
                
                # Determine which items to keep
                kept_items = {s: item for s, item in build.items.items() if s not in actual_order and item is not None}
                
                base_build = build_base_from_items(build.weapon_type, kept_items)
                if base_build is None:
                    continue # Skip if the remaining items cannot be equipped without the replaced ones
                    
                self._dfs(base_build, actual_order, top_j, pool)
                
        return pool.get_builds()
