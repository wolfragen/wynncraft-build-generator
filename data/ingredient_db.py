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
        # Sort ingredients by approximate score-contribution descending so the
        # DFS picks high-impact ingredients first. A high best_score early =
        # more aggressive BB pruning on subsequent branches. Heuristic:
        #   ing_score = sum(w_eff[s] * stat_max[s] | w_eff[s] > 0)
        #             + sum(w_eff[s] * stat_min[s] | w_eff[s] < 0)
        # where w_eff folds direct stat weights AND composite-stat weights
        # propagated to their dep stats (each composite weight contributes to
        # every dep). Crude — composites are non-linear in the deps — but
        # cheap and roughly correct: ingredients high on the heaviest-weighted
        # composite's deps bubble up first.
        # ------------------------------------------------------------
        if self.count > 1:
            w_eff = query.weights_proj.astype(np.float64).copy()
            for c in range(query.comp_count):
                cw = float(query.comp_weight[c])
                if cw == 0.0:
                    continue
                off = int(query.comp_dep_offset[c])
                cnt = int(query.comp_dep_count[c])
                for j in range(off, off + cnt):
                    w_eff[int(query.comp_dep_indices[j])] += cw

            stat_max_f = self.stat_max_matrix.astype(np.float64)
            stat_min_f = self.stat_min_matrix.astype(np.float64)
            pos_w = np.maximum(w_eff, 0.0)
            neg_w = np.minimum(w_eff, 0.0)
            ing_scores = stat_max_f @ pos_w + stat_min_f @ neg_w

            order = np.argsort(-ing_scores, kind="stable")
            self.stat_min_matrix = np.ascontiguousarray(self.stat_min_matrix[order])
            self.stat_max_matrix = np.ascontiguousarray(self.stat_max_matrix[order])
            self.json_ids = np.ascontiguousarray(self.json_ids[order])

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


def build_dual_db(pos_list, neg_list, query):
    """
    Build ONE search DB whose rows are the positive-eff candidates (indices
    [0, pos_count)) followed by the negative-eff candidates ([pos_count, count)).
    Each region is an independent score-sorted `IngredientDB`, so a void slot can
    iterate only the region matching its effectiveness sign (see search_engine).

    The concatenated `json_ids` map void-fill indices straight back to ingredient
    ids with no per-sign bookkeeping in `_update_best`.

    Returns an `IngredientDB`-shaped object exposing `.pos_count` plus the usual
    `stat_min_matrix / stat_max_matrix / json_ids / count / contrib_* masks`.
    """
    db_pos = IngredientDB(pos_list, query)
    db_neg = IngredientDB(neg_list, query)

    db = IngredientDB.__new__(IngredientDB)
    db.stat_count = query.stat_count
    db.pos_count = db_pos.count
    db.count = db_pos.count + db_neg.count

    db.stat_min_matrix = np.ascontiguousarray(
        np.vstack((db_pos.stat_min_matrix, db_neg.stat_min_matrix))
    )
    db.stat_max_matrix = np.ascontiguousarray(
        np.vstack((db_pos.stat_max_matrix, db_neg.stat_max_matrix))
    )
    db.json_ids = np.ascontiguousarray(
        np.concatenate((db_pos.json_ids, db_neg.json_ids))
    )
    db.contrib_pos_mask = db.stat_max_matrix > 0
    db.contrib_neg_mask = db.stat_min_matrix < 0
    return db