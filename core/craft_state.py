import numpy as np

class CraftState:

    __slots__ = ("ingredients", "depth", "max_depth")

    def __init__(self, max_depth):
        self.ingredients = np.zeros(max_depth, dtype=np.int32)
        self.depth = 0
        self.max_depth = max_depth

    def apply(self, ing):
        self.ingredients[self.depth] = ing
        self.depth += 1

    def undo(self):
        self.depth -= 1