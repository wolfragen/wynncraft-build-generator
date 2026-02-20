import numpy as np
from core.grid import LEFT, RIGHT, ABOVE, UNDER, TOUCHING, NOT_TOUCHING


class CraftState:

    __slots__ = (
        "depth",
        "max_depth",
        "durability",
        "slot_raw",
        "slot_effectiveness",
        "slot_contrib",
        "slot_stat_ids",
        "effective_stat_max",
        "ingredients",
        "n_stats",
    )

    def __init__(self, query, recipe, max_depth=6):

        self.n_stats = query.n_stats

        self.slot_raw = np.zeros((6, self.n_stats), dtype=np.int32)
        self.slot_contrib = np.zeros((6, self.n_stats), dtype=np.int32)
        self.slot_stat_ids = [[] for _ in range(6)]

        self.slot_effectiveness = np.zeros(6, dtype=np.int16)
        self.effective_stat_max = np.zeros(self.n_stats, dtype=np.int32)

        self.durability = recipe.scaled_dura_max
        self.depth = 0
        self.max_depth = max_depth
        self.ingredients = np.zeros(max_depth, dtype=np.int32)

    # -------------------------------------------------

    def apply(self, db, ing_idx):

        slot = self.depth
        self.ingredients[slot] = ing_idx

        raw_row = self.slot_raw[slot]
        contrib_row = self.slot_contrib[slot]
        eff_stats = self.effective_stat_max
        eff_array = self.slot_effectiveness

        # ---- Add raw ----
        start = db.stat_ptr[ing_idx]
        end   = db.stat_ptr[ing_idx + 1]

        mult = 100 + eff_array[slot]

        for k in range(start, end):
            stat_id = db.stat_ids[k]

            if raw_row[stat_id] == 0:
                self.slot_stat_ids[slot].append(stat_id)

            raw_row[stat_id] += db.stat_max[k]

            old = contrib_row[stat_id]
            new = (raw_row[stat_id] * mult) // 100

            eff_stats[stat_id] += (new - old)
            contrib_row[stat_id] = new

        self.durability += db.durability[ing_idx]

        # ---- Propagation inline ----
        eff_row = db.effectiveness[ing_idx]
        current_depth = self.depth

        # directional
        for idx, delta in (
            (LEFT[slot],  eff_row[0]),
            (RIGHT[slot], eff_row[1]),
            (ABOVE[slot], eff_row[2]),
            (UNDER[slot], eff_row[3]),
        ):
            if idx is not None and idx < current_depth and delta != 0:

                old_mult = 100 + eff_array[idx]
                new_mult = old_mult + delta
                eff_array[idx] += delta

                raw_r = self.slot_raw[idx]
                contrib_r = self.slot_contrib[idx]
                stat_ids = self.slot_stat_ids[idx]

                for stat_id in stat_ids:
                    raw = raw_r[stat_id]
                    old = contrib_r[stat_id]
                    new = (raw * new_mult) // 100
                    eff_stats[stat_id] += (new - old)
                    contrib_r[stat_id] = new

        # touching
        delta = eff_row[4]
        if delta != 0:
            for idx in TOUCHING[slot]:
                if idx < current_depth:
                    old_mult = 100 + eff_array[idx]
                    new_mult = old_mult + delta
                    eff_array[idx] += delta

                    raw_r = self.slot_raw[idx]
                    contrib_r = self.slot_contrib[idx]
                    stat_ids = self.slot_stat_ids[idx]

                    for stat_id in stat_ids:
                        raw = raw_r[stat_id]
                        old = contrib_r[stat_id]
                        new = (raw * new_mult) // 100
                        eff_stats[stat_id] += (new - old)
                        contrib_r[stat_id] = new

        # not touching
        delta = eff_row[5]
        if delta != 0:
            for idx in NOT_TOUCHING[slot]:
                if idx < current_depth:
                    old_mult = 100 + eff_array[idx]
                    new_mult = old_mult + delta
                    eff_array[idx] += delta

                    raw_r = self.slot_raw[idx]
                    contrib_r = self.slot_contrib[idx]
                    stat_ids = self.slot_stat_ids[idx]

                    for stat_id in stat_ids:
                        raw = raw_r[stat_id]
                        old = contrib_r[stat_id]
                        new = (raw * new_mult) // 100
                        eff_stats[stat_id] += (new - old)
                        contrib_r[stat_id] = new

        self.depth += 1

    # -------------------------------------------------

    def undo(self, db):

        self.depth -= 1
        slot = self.depth
        ing_idx = self.ingredients[slot]

        raw_row = self.slot_raw[slot]
        contrib_row = self.slot_contrib[slot]
        eff_stats = self.effective_stat_max
        eff_array = self.slot_effectiveness

        # ---- Remove propagation inline ----
        eff_row = db.effectiveness[ing_idx]
        current_depth = self.depth

        for idx, delta in (
            (LEFT[slot],  eff_row[0]),
            (RIGHT[slot], eff_row[1]),
            (ABOVE[slot], eff_row[2]),
            (UNDER[slot], eff_row[3]),
        ):
            if idx is not None and idx < current_depth and delta != 0:

                old_mult = 100 + eff_array[idx]
                new_mult = old_mult - delta
                eff_array[idx] -= delta

                raw_r = self.slot_raw[idx]
                contrib_r = self.slot_contrib[idx]
                stat_ids = self.slot_stat_ids[idx]

                for stat_id in stat_ids:
                    raw = raw_r[stat_id]
                    old = contrib_r[stat_id]
                    new = (raw * new_mult) // 100
                    eff_stats[stat_id] += (new - old)
                    contrib_r[stat_id] = new

        # touching
        delta = eff_row[4]
        if delta != 0:
            for idx in TOUCHING[slot]:
                if idx < current_depth:
                    old_mult = 100 + eff_array[idx]
                    new_mult = old_mult - delta
                    eff_array[idx] -= delta

                    raw_r = self.slot_raw[idx]
                    contrib_r = self.slot_contrib[idx]
                    stat_ids = self.slot_stat_ids[idx]

                    for stat_id in stat_ids:
                        raw = raw_r[stat_id]
                        old = contrib_r[stat_id]
                        new = (raw * new_mult) // 100
                        eff_stats[stat_id] += (new - old)
                        contrib_r[stat_id] = new

        # not touching
        delta = eff_row[5]
        if delta != 0:
            for idx in NOT_TOUCHING[slot]:
                if idx < current_depth:
                    old_mult = 100 + eff_array[idx]
                    new_mult = old_mult - delta
                    eff_array[idx] -= delta

                    raw_r = self.slot_raw[idx]
                    contrib_r = self.slot_contrib[idx]
                    stat_ids = self.slot_stat_ids[idx]

                    for stat_id in stat_ids:
                        raw = raw_r[stat_id]
                        old = contrib_r[stat_id]
                        new = (raw * new_mult) // 100
                        eff_stats[stat_id] += (new - old)
                        contrib_r[stat_id] = new

        # ---- Remove raw ----
        start = db.stat_ptr[ing_idx]
        end   = db.stat_ptr[ing_idx + 1]

        mult = 100 + eff_array[slot]

        for k in range(start, end):
            stat_id = db.stat_ids[k]

            old = contrib_row[stat_id]

            raw_row[stat_id] -= db.stat_max[k]

            new = (raw_row[stat_id] * mult) // 100 if raw_row[stat_id] != 0 else 0

            eff_stats[stat_id] += (new - old)
            contrib_row[stat_id] = new

        self.durability -= db.durability[ing_idx]

        raw_row.fill(0)
        contrib_row.fill(0)
        self.slot_stat_ids[slot].clear()