from data_loader import load_game_data
from scoring import *
from search import generate_builds, replace_item_in_build

def main():
    print("Loading game data...")
    items_database, sets_database = load_game_data('items.json')

    orders_to_test = [
        ["Chestplate", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Wand"],
        ["Wand", "Leggings", "Helmet", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace", "Chestplate"]
    ]
    
    score_build_function = score_build_spellDmg
    score_item_function = score_item_spellDmg
    
    print("Generating initial build...")
    best_build = generate_builds(
        items_database=items_database,
        score_build_function=score_build_function,
        score_item_function=score_item_function,
        imposed_items=[("Wand", "Depressing Stick")],
        weapon_type="Wand",
        equip_orders=orders_to_test,
        top_k=1
    )
    
    replace_order = [ "Wand", "Helmet", "Chestplate", "Leggings", "Boots", "Ring1", "Ring2", "Bracelet", "Necklace"]
    # try to replace each piece in the order defined above, if we find a better one we update the build and continue replacing the next pieces with the new build until we have tested all pieces or we can't find any better one
    for slot in replace_order:
        print(f"Trying to replace {slot}...")
        new_build = replace_item_in_build(
            build=best_build,
            slot_to_replace=slot,
            items_database=items_database,
            score_build_function=score_build_function,
            score_item_function=score_item_function,
            top_k=1
        )
        if new_build is not None and score_build_function(new_build) > score_build_function(best_build):
            print(f"Found better build by replacing {slot}!")
            best_build = new_build
        else:
            print(f"No better build found by replacing {slot}.")

    # print final build and its score
    print("Best build found:")
    print(f" - ".join(slot + ": " + item['displayName'] for slot, item in best_build.items.items() if item is not None))
    print(f"Available SP: {best_build.available_skill_points}")
    print(f"Stats: {best_build.stats}")
    print(f"Score: {score_build_function(best_build)}")



if __name__ == "__main__":
    main()

# https://wynnbuilder.github.io/builder/#CN0qJmW7me0yQG9AAH1LEM10i6oxnFgcD-E
# exemple build with random arcanist ability tree to test generated builds