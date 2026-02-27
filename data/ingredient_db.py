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
            filtered_ingredients: list of RawIngredient
            query: Query instance
        """

        self.count = len(filtered_ingredients)
        self.stat_count = query.stat_count
        active_indices = query.active_indices

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

            self.stat_min_matrix[new_idx] = ing.stats_min[active_indices]
            self.stat_max_matrix[new_idx] = ing.stats_max[active_indices]
            self.json_ids[new_idx] = ing.ing_id

        # ------------------------------------------------------------
        # Contribution masks [N, K]
        # ------------------------------------------------------------

        # Positive contribution (normal search)
        self.contrib_pos_mask = self.stat_max_matrix > 0

        # Negative contribution (used for inversion search)
        self.contrib_neg_mask = self.stat_min_matrix < 0

    # ------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------

    def get_json_id(self, idx: int):
        return self.json_ids[idx]

    def __len__(self):
        return self.count