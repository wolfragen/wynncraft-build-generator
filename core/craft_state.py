import numpy as np

class CraftState:
    """
    Represents a whole craft
    TODO: 
        - depth might need to be changed as len(ingredients)
        - We will probably need to store which stats exists in the current craft to see which ones to add to make it valid
    """

    __slots__ = ("ingredients", "depth", "max_depth")

    def __init__(self, max_depth):
        self.ingredients = np.zeros(max_depth, dtype=np.int32)
        self.depth = 0
        self.max_depth = max_depth

    def apply(self, ing): # Add another ingredients
        self.ingredients[self.depth] = ing
        self.depth += 1

    def undo(self): # "Remove" one ingredients (it'll be overriden next apply)
        self.depth -= 1