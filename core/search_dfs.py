from core.search_base import SearchBase


class DFSSearch(SearchBase):

    def run(self, state):
        self._dfs(state)

    def _dfs(self, state):

        if not self.prune(state):
            pass
        else:
            return

        if state.depth == state.max_depth:

            if self.is_solution(state):
                self.evaluate(state)

            return

        for ing in range(self.db.n):

            state.apply(self.db, ing)

            self._dfs(state)

            state.undo(self.db)