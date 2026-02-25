"""
ingredient_db.py

Builds compact ingredient database after filtering.

Design goals:
- Store projected min/max stat matrices
- Store only JSON id
- Precompute contributor structures
- Optimized for search phase
"""

import numpy as np


class IngredientDB:
    """
    Compact ingredient database optimized for search.
    """

    def __init__(self, filtered_ingredients: list, query):
        """
        Build optimized database from filtered ingredients.

        Args:
            filtered_ingredients: list of raw ingredient dicts
            query: Query instance
        """

        self.count = len(filtered_ingredients)
        self.stat_count = query.stat_count
        active_indices = query.active_indices
        search_inv = query.search_for_inversion

        # ------------------------------------------------------------
        # Projected stat matrices [N, K]
        # ------------------------------------------------------------
        self.stat_min_matrix = np.empty(
            (self.count, self.stat_count),
            dtype=np.int16
        )

        self.stat_max_matrix = np.empty(
            (self.count, self.stat_count),
            dtype=np.int16
        )

        # ------------------------------------------------------------
        # JSON id mapping
        # ------------------------------------------------------------
        self.json_ids = np.empty(self.count, dtype=np.int32)

        for new_idx, ing in enumerate(filtered_ingredients):

            self.stat_min_matrix[new_idx] = ing["stats_min"][active_indices]
            self.stat_max_matrix[new_idx] = ing["stats_max"][active_indices]
            self.json_ids[new_idx] = ing["id"]

        # ------------------------------------------------------------
        # Contribution mask [N, K]
        # ------------------------------------------------------------
        if not search_inv:
            # Normal search: stat can be positive
            self.contrib_mask = self.stat_max_matrix > 0
        else:
            # Inversion search: stat can be negative
            self.contrib_mask = self.stat_min_matrix < 0

        # ------------------------------------------------------------
        # Contributors per stat
        # ------------------------------------------------------------
        self.stat_contributors = [
            np.nonzero(self.contrib_mask[:, k])[0].astype(np.int32)
            for k in range(self.stat_count)
        ]

        # ------------------------------------------------------------
        # Bitmask per ingredient
        # ------------------------------------------------------------
        self.stat_bitmask = np.zeros(self.count, dtype=np.uint64)

        for k in range(self.stat_count):
            bit = np.uint64(1) << np.uint64(k)
            self.stat_bitmask[self.contrib_mask[:, k]] |= bit

    # ------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------

    def get_json_id(self, idx: int):
        return self.json_ids[idx]

    def get_bitmask(self, idx: int):
        return self.stat_bitmask[idx]

    def __len__(self):
        return self.count