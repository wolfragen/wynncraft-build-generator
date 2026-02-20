CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-"

ATTACK_SPEED_MAP = {
    "SUPER_SLOW": 0,
    "VERY_SLOW": 1,
    "SLOW": 2,
    "NORMAL": 3,
    "FAST": 4,
    "VERY_FAST": 5,
    "SUPER_FAST": 6,
}


class EncodingBitVector:
    __slots__ = ("value", "length")

    def __init__(self):
        self.value = 0
        self.length = 0

    def append(self, v: int, bitlen: int):
        self.value |= (v << self.length)
        self.length += bitlen

    def pad_to_6bits(self):
        padding = (-self.length) % 6
        if padding:
            self.append(0, padding)

    def finalize(self) -> str:
        self.pad_to_6bits()
        out = []
        temp = self.value
        for _ in range(self.length // 6):
            out.append(CHARSET[temp & 0b111111])
            temp >>= 6
        return "".join(out)


def generate_crafter_url(
    recipe_json_id: int,
    tier: int,
    ingredient_json_ids: list[int],
    raw_ingredients: list[dict],
    raw_recipes: list[dict],
    atk_speed: str = "NORMAL",  # must be full name
) -> str:

    if len(ingredient_json_ids) != 6:
        raise ValueError("Crafter requires exactly 6 ingredients.")

    ing_id_to_index = {ing["id"]: i for i, ing in enumerate(raw_ingredients)}
    recipe_id_to_index = {r["id"]: i for i, r in enumerate(raw_recipes)}

    ing_indices = [ing_id_to_index[i] for i in ingredient_json_ids]
    recipe_index = recipe_id_to_index[recipe_json_id]

    vec = EncodingBitVector()

    # legacy bit
    vec.append(0, 1)

    # version (2)
    vec.append(2, 7)

    # ingredients
    for idx in ing_indices:
        vec.append(idx, 12)

    # recipe
    vec.append(recipe_index, 12)

    # materials (tier-1)
    mat_val = tier - 1
    vec.append(mat_val, 3)
    vec.append(mat_val, 3)

    # weapon attack speed
    is_weapon = raw_recipes[recipe_index]["type"] == "weapon"

    if is_weapon:
        vec.append(ATTACK_SPEED_MAP[atk_speed], 4)

    return f"https://wynnbuilder.github.io/crafter/#{vec.finalize()}"