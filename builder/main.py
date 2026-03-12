from data_loader import load_game_data
from scoring import *
from search import UnifiedBuilder

def print_build(build, score, rank):
    print(f"\\n--- Rank {rank} Build (Score: {score:.2f}) ---")
    for slot, item in build.items.items():
        if item:
            print(f"{slot}: {item['displayName']}")
    print(f"Available SP: {build.available_skill_points}")
    print(f"Stats: {build.stats}")

def main():
    print("Loading game data...")
    items_database, sets_database = load_game_data('items.json')

    builder = UnifiedBuilder(
        items_database=items_database,
        score_build_fn=score_build_spellDmg,
        score_item_fn=score_item_spellDmg
    )

    imposed_items = [("Wand", "Pure")]
    weapon_type = "Wand"
    
    fill_orders = [
        ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Wand"],
        ["Wand", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Chestplate"]
    ]
    
    top_k = 3 # number of best items to create tree branches on
    top_i = 3 # number of best builds kept
    
    print(f"\\n--- Phase 1: Generating Top {top_i} Builds ---")
    best_builds = builder.generate(
        weapon_type=weapon_type,
        imposed_items=imposed_items,
        fill_orders=fill_orders,
        top_k=top_k,
        top_i=top_i
    )
    
    replace_orders = [
        ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Wand"],
        ["Wand", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Chestplate"]
        #["Helmet", "Chestplate"],
        #["Ring1", "Ring2", "Bracelet", "Necklace"]
    ]
    imposed_slots = [slot for slot, _ in imposed_items]
    
    top_j = 3 # number of best replacements to explore down the tree
    top_l = 3 # final number of best builds to keep
    
    print(f"\\n--- Phase 2: Refining Builds (Top {top_l} final) ---")
    final_builds = builder.refine(
        builds=best_builds,
        replace_orders=replace_orders,
        imposed_slots=imposed_slots,
        top_j=top_j,
        top_l=top_l
    )
    
    print(f"\\n--- Final Top {top_l} Builds ---")
    for rank, (score, build) in enumerate(final_builds, 1):
        print_build(build, score, rank)

if __name__ == "__main__":
    main()
