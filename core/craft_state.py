import numpy as np
from core.grid import LEFT, RIGHT, ABOVE, UNDER, TOUCHING, NOT_TOUCHING


class CraftState:

    __slots__ = (
        "slot_raw_min",
        "slot_raw_max",
        "durability",
        "depth",
        "max_depth",

        "slots",
        "slot_effectiveness",

        "ingredients",
        "n_stats",
    )

    def __init__(self, query, recipe, max_depth=6):

        self.n_stats = query.n_stats
    
        self.slot_raw_min = np.zeros((6, self.n_stats), dtype=np.int32)
        self.slot_raw_max = np.zeros((6, self.n_stats), dtype=np.int32)
    
        # Start from scaled base durability
        self.durability = recipe.scaled_dura_max
    
        self.depth = 0
        self.max_depth = max_depth
    
        self.slots = np.full(6, -1, dtype=np.int32)
        self.slot_effectiveness = np.zeros(6, dtype=np.int32)
    
        self.ingredients = np.zeros(max_depth, dtype=np.int32)

    # -------------------------------------------------

    def apply(self, db, ing_idx):

        slot = self.depth

        self.slots[slot] = ing_idx
        self.ingredients[self.depth] = ing_idx
        self.depth += 1

        # ---- Store raw stats in slot ----
        start = db.stat_ptr[ing_idx]
        end   = db.stat_ptr[ing_idx + 1]

        for k in range(start, end):
            stat_id = db.stat_ids[k]
            self.slot_raw_min[slot, stat_id] += db.stat_min[k]
            self.slot_raw_max[slot, stat_id] += db.stat_max[k]

        # ---- Durability ----
        self.durability += db.durability[ing_idx]

        # ---- Effectiveness propagation ----
        eff = db.effectiveness[ing_idx]

        if eff[0] != 0 and LEFT[slot] is not None:
            self.slot_effectiveness[LEFT[slot]] += eff[0]

        if eff[1] != 0 and RIGHT[slot] is not None:
            self.slot_effectiveness[RIGHT[slot]] += eff[1]

        if eff[2] != 0 and ABOVE[slot] is not None:
            self.slot_effectiveness[ABOVE[slot]] += eff[2]

        if eff[3] != 0 and UNDER[slot] is not None:
            self.slot_effectiveness[UNDER[slot]] += eff[3]

        if eff[4] != 0:
            for s in TOUCHING[slot]:
                self.slot_effectiveness[s] += eff[4]

        if eff[5] != 0:
            for s in NOT_TOUCHING[slot]:
                self.slot_effectiveness[s] += eff[5]

    # -------------------------------------------------

    def undo(self, db):

        self.depth -= 1
        slot = self.depth
        ing_idx = self.slots[slot]

        # ---- Remove raw stats ----
        start = db.stat_ptr[ing_idx]
        end   = db.stat_ptr[ing_idx + 1]

        for k in range(start, end):
            stat_id = db.stat_ids[k]
            self.slot_raw_min[slot, stat_id] -= db.stat_min[k]
            self.slot_raw_max[slot, stat_id] -= db.stat_max[k]

        # ---- Durability ----
        self.durability -= db.durability[ing_idx]

        # ---- Remove effectiveness ----
        eff = db.effectiveness[ing_idx]

        if eff[0] != 0 and LEFT[slot] is not None:
            self.slot_effectiveness[LEFT[slot]] -= eff[0]

        if eff[1] != 0 and RIGHT[slot] is not None:
            self.slot_effectiveness[RIGHT[slot]] -= eff[1]

        if eff[2] != 0 and ABOVE[slot] is not None:
            self.slot_effectiveness[ABOVE[slot]] -= eff[2]

        if eff[3] != 0 and UNDER[slot] is not None:
            self.slot_effectiveness[UNDER[slot]] -= eff[3]

        if eff[4] != 0:
            for s in TOUCHING[slot]:
                self.slot_effectiveness[s] -= eff[4]

        if eff[5] != 0:
            for s in NOT_TOUCHING[slot]:
                self.slot_effectiveness[s] -= eff[5]

        self.slots[slot] = -1

    # -------------------------------------------------

    def compute_effective_stat(self, stat_id):
        """
        Exact effective stat computation.
        """

        total = 0.0

        for s in range(self.depth):
            raw = self.slot_raw_max[s, stat_id]
            amp = 1.0 + self.slot_effectiveness[s] / 100.0
            total += raw * amp

        return total