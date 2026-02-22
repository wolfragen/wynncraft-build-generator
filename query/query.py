"""
query.py

Handles stat query parsing and projection.

Design goals:
- Support all stats defined in data.stats
- Preserve existing mask semantics
- Expose full stat space arrays for filtering
- Provide projected stat space for search
- No dict lookups in hot path
"""

import numpy as np

from data.stats import STAT_INDEX, STAT_COUNT


class Query:
    """
    Represents a single stat query.
    """

    def __init__(
        self,
        user_json: dict,
        search_for_inversion: bool,
        item_type=None,
        skill=None,
    ):
        """
        Parse user query.

        Args:
            user_json: dict of stat constraints
            search_for_inversion: bool
            item_type: optional crafting profession filter
            skill: optional skill filter (reserved for later use)
        """

        # ------------------------------------------------------------
        # Store flags used by ingredient_filter
        # ------------------------------------------------------------
        self.search_for_inversion = search_for_inversion
        self.item_type = item_type
        self.skill = skill

        self.min_durability = None

        # ------------------------------------------------------------
        # Full stat space storage (used by filter)
        # ------------------------------------------------------------
        self.min = np.zeros(STAT_COUNT, dtype=np.int32)
        self.max = np.zeros(STAT_COUNT, dtype=np.int32)
        self.weights = np.zeros(STAT_COUNT, dtype=np.float32)

        self.has_min_mask = np.zeros(STAT_COUNT, dtype=np.bool_)
        self.has_max_mask = np.zeros(STAT_COUNT, dtype=np.bool_)

        active_mask = np.zeros(STAT_COUNT, dtype=np.bool_)

        # ------------------------------------------------------------
        # Parse user JSON
        # ------------------------------------------------------------
        for stat_name, config in user_json.items():

            # Special durability handling
            if stat_name == "durability":
                self.min_durability = config.get("min")
                continue

            idx = STAT_INDEX.get(stat_name)
            if idx is None:
                continue

            stat_min = config.get("min")
            stat_max = config.get("max")
            stat_weight = config.get("weight")

            if stat_min is not None:
                self.min[idx] = stat_min
                self.has_min_mask[idx] = True
                active_mask[idx] = True

            if stat_max is not None:
                self.max[idx] = stat_max
                self.has_max_mask[idx] = True
                active_mask[idx] = True

            if stat_weight is not None:
                self.weights[idx] = stat_weight
                active_mask[idx] = True

        # ------------------------------------------------------------
        # Build projected stat space (for search phase)
        # ------------------------------------------------------------
        self.active_indices = np.nonzero(active_mask)[0].astype(np.int32)
        self.stat_count = len(self.active_indices)

        self.min_proj = self.min[self.active_indices]
        self.max_proj = self.max[self.active_indices]
        self.weights_proj = self.weights[self.active_indices]

        self.has_min_mask_proj = self.has_min_mask[self.active_indices]
        self.has_max_mask_proj = self.has_max_mask[self.active_indices]

        self.weight_mask_proj = self.weights_proj != 0.0

    # ------------------------------------------------------------
    # Projection helper
    # ------------------------------------------------------------

    def project_stat_matrix(self, stat_matrix: np.ndarray) -> np.ndarray:
        """
        Project ingredient stat matrix into active stat space.

        Input:
            stat_matrix: [N, STAT_COUNT]

        Output:
            [N, active_stat_count]
        """
        return stat_matrix[:, self.active_indices]