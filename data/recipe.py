import math

TIER_MULT = [0.0, 1.0, 1.25, 1.4]


class Recipe:

    __slots__ = (
        "base_dura_min",
        "base_dura_max",
        "scaled_dura_min",
        "scaled_dura_max",
    )

    def __init__(self, recipe_data, tier):

        self.base_dura_min = recipe_data["durability"]["minimum"]
        self.base_dura_max = recipe_data["durability"]["maximum"]

        mult = TIER_MULT[tier]

        # Apply material multiplier exactly like craft.js
        self.scaled_dura_min = math.floor(self.base_dura_min * mult)
        self.scaled_dura_max = math.floor(self.base_dura_max * mult)