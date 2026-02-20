import numpy as np


class IngredientDB:
    """
    Compact ingredient database aligned with global stat_index.

    Stores:
    - ingredient JSON id
    - CSR stat representation (global stat ids)
    - durability (not affected by effectiveness)
    - effectiveness modifiers (6 directions)
    """

    __slots__ = (
        "n",
        "ids",
        "stat_ptr",
        "stat_ids",
        "stat_min",
        "stat_max",
        "durability",
        "effectiveness",
    )

    def __init__(self, ingredients_raw, stat_index):

        self.n = len(ingredients_raw)
        self.ids = np.zeros(self.n, dtype=np.int32)

        # ---------- Count total stat entries ----------
        total_entries = 0
        for ing in ingredients_raw:
            ids = ing.get("ids")
            if ids:
                total_entries += len(ids)

        # ---------- Allocate CSR ----------
        self.stat_ptr = np.zeros(self.n + 1, dtype=np.int32)
        self.stat_ids = np.zeros(total_entries, dtype=np.uint16)
        self.stat_min = np.zeros(total_entries, dtype=np.int32)
        self.stat_max = np.zeros(total_entries, dtype=np.int32)

        self.durability = np.zeros(self.n, dtype=np.int32)
        self.effectiveness = np.zeros((self.n, 6), dtype=np.int16)

        # ---------- Fill ----------
        cursor = 0

        for i, ing in enumerate(ingredients_raw):

            self.stat_ptr[i] = cursor
            self.ids[i] = ing["id"]

            ids = ing.get("ids")
            if ids:
                for stat_name, data in ids.items():

                    if stat_name not in stat_index:
                        continue

                    gid = stat_index[stat_name]

                    if isinstance(data, dict):
                        min_val = data.get("minimum", data.get("min", 0))
                        max_val = data.get("maximum", data.get("max", 0))
                    else:
                        min_val = data
                        max_val = data

                    self.stat_ids[cursor] = gid
                    self.stat_min[cursor] = min_val
                    self.stat_max[cursor] = max_val

                    cursor += 1

            # Durability (NOT affected by effectiveness)
            self.durability[i] = ing.get("itemIDs", {}).get("dura", 0)

            # Effectiveness
            pos = ing.get("posMods", {})
            self.effectiveness[i, 0] = pos.get("left", 0)
            self.effectiveness[i, 1] = pos.get("right", 0)
            self.effectiveness[i, 2] = pos.get("above", 0)
            self.effectiveness[i, 3] = pos.get("under", 0)
            self.effectiveness[i, 4] = pos.get("touching", 0)
            self.effectiveness[i, 5] = pos.get("notTouching", 0)

        self.stat_ptr[self.n] = cursor