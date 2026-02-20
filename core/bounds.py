import numpy as np


class GlobalBounds:

    __slots__ = (
        "max_stat_gain",
        "min_stat_gain",
        "max_durability",
        "min_durability",
        "max_effectiveness",
        "min_effectiveness",
    )

    def __init__(self, db, query):

        n = query.n_stats

        self.max_stat_gain = np.zeros(n, dtype=np.int32)
        self.min_stat_gain = np.zeros(n, dtype=np.int32)

        max_eff = 0
        min_eff = 0

        for ing in range(db.n):

            # ----- Stats -----
            start = db.stat_ptr[ing]
            end   = db.stat_ptr[ing + 1]

            for k in range(start, end):
                stat_id = db.stat_ids[k]

                max_val = db.stat_max[k]
                min_val = db.stat_min[k]

                if max_val > self.max_stat_gain[stat_id]:
                    self.max_stat_gain[stat_id] = max_val

                if min_val < self.min_stat_gain[stat_id]:
                    self.min_stat_gain[stat_id] = min_val

            # ----- Effectiveness -----
            eff_row = db.effectiveness[ing]
            local_max = eff_row.max()
            local_min = eff_row.min()
            
            if local_max > max_eff:
                max_eff = local_max
            
            if local_min < min_eff:
                min_eff = local_min

        self.max_durability = db.durability.max()
        self.min_durability = db.durability.min()

        self.max_effectiveness = max_eff
        self.min_effectiveness = min_eff
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        