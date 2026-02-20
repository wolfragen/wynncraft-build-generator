from core.craft_state import CraftState


class SearchBase:
    """
    Simplified search engine.

    Leaf-only evaluation model:
    - DFS only selects ingredients
    - All craft computation happens at depth == max_depth
    """

    __slots__ = (
        "db",
        "query",
        "recipe",
        "best_score",
        "best_solution",
    )

    # -------------------------------------------------
    # Entry point
    # -------------------------------------------------

    @staticmethod
    def execute(db, query, recipe, max_depth):

        state = CraftState(max_depth=max_depth)

        from core.search_dfs import DFSSearch
        search = DFSSearch(db, query, recipe)

        search.run(state)

        return search.best_score, search.best_solution

    # -------------------------------------------------

    def __init__(self, db, query, recipe):

        self.db = db
        self.query = query
        self.recipe = recipe

        self.best_score = float("-inf")
        self.best_solution = None