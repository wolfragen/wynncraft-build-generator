"""
milp/solve.py

Run CP-SAT on a built MilpModel and decode the slot assignment.

Status mapping (milp/README.md §6):
  OPTIMAL    — proven best (the only status that may be called "optimal")
  FEASIBLE   — found a solution but hit the time limit; not proven
  INFEASIBLE — no craft satisfies the hard min/max (a valid answer, not an error)
  MODEL_INVALID / UNKNOWN — surface the error
"""

from ortools.sat.python import cp_model


def solve_model(mm, max_time_s=30.0, workers=8, log=False):
    """
    Returns a dict:
      status, status_name, chosen_rows (6 db-row indices in slot order, or None),
      objective, best_bound, solver.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_s)
    solver.parameters.num_search_workers = int(workers)
    if log:
        solver.parameters.log_search_progress = True

    status = solver.Solve(mm.model)

    chosen_rows = None
    objective = None
    best_bound = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen_rows = []
        for p in range(6):
            sel = -1
            for i in range(mm.N):
                if solver.Value(mm.y[p][i]) == 1:
                    sel = i
                    break
            chosen_rows.append(sel)
        objective = solver.ObjectiveValue()
        best_bound = solver.BestObjectiveBound()

    return {
        "status": status,
        "status_name": solver.StatusName(status),
        "chosen_rows": chosen_rows,
        "objective": objective,
        "best_bound": best_bound,
        "solver": solver,
    }
