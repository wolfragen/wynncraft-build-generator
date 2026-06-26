"""
milp — exact mixed-integer (CP-SAT) solver track for the crafting optimizer.

A second, independent solver alongside the heuristic DFS in core/search_engine.py.
It reuses the shared data layer and the craft.js oracle and adds only the
optimisation model. See README.md for the full design.
"""
