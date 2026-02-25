import json

def contains_permutation(json_path, target_list):
    if len(target_list) != 6:
        raise ValueError("target_list must contain exactly 6 elements")

    target_sorted = sorted(target_list)

    with open(json_path, "r") as f:
        data = json.load(f)
    found = False
    perms = []

    for entry in data:
        if "ings" in entry and len(entry["ings"]) == 6:
            if sorted(entry["ings"]) == target_sorted:
                found = True
                perms.append(entry["ings"])

    return found, perms


if __name__ == "__main__":
    json_file = "data/precalc/full/JEWELING_META_4.json"
    target = [756, -1, 756, -1, 359, 614]

    found, perm = contains_permutation(json_file, target)

    if found:
        print("Permutation found!")
        print(perm)
    else:
        print("No permutation found.")
        