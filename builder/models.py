import copy
from data_loader import skill_point_types, maximized_stats, build_unique_stats, item_only_stats, stat_to_max

class BuildInfo:
    def __init__(self, weapon_type, items=None, skill_points=None, available_skill_points=None, max_reqs=None, exclusive_flags=None):
        self.weapon_type = weapon_type
        self.items = items or {
            weapon_type: None, "Helmet": None, "Chestplate": None, 
            "Leggings": None, "Boots": None, "Ring1": None, 
            "Ring2": None, "Bracelet": None, "Necklace": None
        }
        self.available_skill_points = available_skill_points if available_skill_points is not None else 200
        self.skill_points = skill_points or {sk: {"attributed": 0, "items_counting": 0, "items_not_counting": 0} for sk in skill_point_types}
        self.max_reqs = max_reqs or {sk: -150 for sk in skill_point_types}
        self.exclusive_flags = exclusive_flags or set()
        self.stats = {}

    def add_item(self, slot_name, item_info):
        item, new_attributions, new_available_skill_points = item_info
        new_items = self.items.copy()
        new_items[slot_name] = item
        
        new_sp = copy.deepcopy(self.skill_points)
        new_flags = self.exclusive_flags.copy()
        
        new_build = BuildInfo(self.weapon_type, items=new_items, skill_points=new_sp, exclusive_flags=new_flags)
        new_build.max_reqs = self.max_reqs.copy()
        
        for sk in skill_point_types:
            skReq = sk + "Req"
            if skReq in item and item[skReq] > new_build.max_reqs[sk]:
                new_build.max_reqs[sk] = item[skReq]

        new_build.available_skill_points = new_available_skill_points
        for sk in new_attributions.keys():
            new_build.skill_points[sk]["attributed"] += new_attributions[sk]

        for sk in skill_point_types:
            if sk in item:
                if item.get("category") == "weapon":
                    new_build.skill_points[sk]["items_not_counting"] += stat_to_max(item[sk])
                else:
                    new_build.skill_points[sk]["items_counting"] += item[sk]
        
        return new_build
    
    def calculate_stats(self):
        self.stats = {sk: self.skill_points[sk]["attributed"] + self.skill_points[sk]["items_counting"] + self.skill_points[sk]["items_not_counting"] for sk in skill_point_types}
        self.stats["sdRaw"] = 0
        self.stats["sdPct"] = 0
        for item in self.items.values():
            if item is not None:
                for stat in item:
                    if stat in maximized_stats:
                        if stat not in self.stats:
                            self.stats[stat] = stat_to_max(item[stat])
                        else:
                            self.stats[stat] = max(self.stats[stat], stat_to_max(item[stat]))
                    elif stat in build_unique_stats:
                        self.stats[stat] = item[stat]
                    elif stat not in item_only_stats:
                        if stat not in self.stats:
                            self.stats[stat] = stat_to_max(item[stat])
                        else:
                            self.stats[stat] += stat_to_max(item[stat])
