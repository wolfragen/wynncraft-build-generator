from core.search_base import SearchBase
from core.leaf_evaluator import evaluate_leaf
from core.grid import LEFT, RIGHT, ABOVE, UNDER, TOUCHING, NOT_TOUCHING


class DFSSearch(SearchBase):

    __slots__ = ("min_dura",)

    def __init__(self, db, query, recipe):
        super().__init__(db, query, recipe)
        self.min_dura = query.min_durability if query.min_durability is not None else -1

    def run(self, state):
        # Warmup compilation
        evaluate_leaf(
            state.ingredients,
            self.db.effectiveness,
            self.db.stat_ptr,
            self.db.stat_ids,
            self.db.stat_max,
            self.db.durability,
            self.recipe.scaled_dura_max,
            self.query.relevant_stat_indices,
            self.query.has_min_mask,
            self.query.has_max_mask,
            self.query.min_vals,
            self.query.max_vals,
            self.query.weights,
            self.min_dura,
            self.query.durability_weight,
            LEFT, RIGHT, ABOVE, UNDER,
            TOUCHING, NOT_TOUCHING,
        )

        self._dfs(state)

    def _dfs(self, state):

        if state.depth == state.max_depth:

            score = evaluate_leaf(
                state.ingredients,
                self.db.effectiveness,
                self.db.stat_ptr,
                self.db.stat_ids,
                self.db.stat_max,
                self.db.durability,
                self.recipe.scaled_dura_max,
                self.query.relevant_stat_indices,
                self.query.has_min_mask,
                self.query.has_max_mask,
                self.query.min_vals,
                self.query.max_vals,
                self.query.weights,
                self.min_dura,
                self.query.durability_weight,
                LEFT, RIGHT, ABOVE, UNDER,
                TOUCHING, NOT_TOUCHING,
            )

            if score >= 0.0 and score > self.best_score:
                self.best_score = score
                self.best_solution = state.ingredients.copy()

            return

        for ing in range(self.db.n):
            state.apply(ing)
            self._dfs(state)
            state.undo()