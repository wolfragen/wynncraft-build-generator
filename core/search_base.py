from core.bounds import GlobalBounds
from core.craft_state import CraftState
from core.pruning import can_still_reach_requirements
from core.scoring import compute_score


class SearchBase:
    """
    Base search engine.

    Handles:
    - pruning
    - scoring
    - best solution tracking
    - centralized execution entrypoint
    """

    __slots__ = (
        "db",
        "query",
        "bounds",
        "best_score",
        "best_solution",
    )

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------

    @staticmethod
    def execute(db, query, recipe, max_depth):
        bounds = GlobalBounds(db, query)
        state = CraftState(query, recipe, max_depth)
    
        if query.algorithm == "dfs":
            from core.search_dfs import DFSSearch
            search = DFSSearch(db, query, bounds)
        else:
            raise ValueError("Unknown algorithm")
    
        search.run(state)
    
        return search.best_score, search.best_solution

    # -------------------------------------------------

    def __init__(self, db, query, bounds):

        self.db = db
        self.query = query
        self.bounds = bounds

        self.best_score = float("-inf")
        self.best_solution = None

    # -------------------------------------------------

    def is_solution(self, state):

        for i in self.query.relevant_stat_indices:
    
            effective = state.compute_effective_stat(i)
    
            if self.query.has_min_mask[i]:
                if effective < self.query.min_vals[i]:
                    return False
    
            if self.query.has_max_mask[i]:
                if effective > self.query.max_vals[i]:
                    return False
    
        if self.query.min_durability is not None:
            if state.durability < self.query.min_durability:
                return False
    
        return True

    # -------------------------------------------------

    def evaluate(self, state):

        score = compute_score(state, self.query)

        if score > self.best_score:
            self.best_score = score
            self.best_solution = state.ingredients.copy()

    # -------------------------------------------------

    def prune(self, state):
        return False
        return not can_still_reach_requirements(
            state,
            self.query,
            self.bounds,
            best_score=self.best_score
        )
    
    
    
    
    
    
    
    
    
    
    
    
    