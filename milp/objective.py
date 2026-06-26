"""
milp/objective.py

The objective SEAM. v1 reproduces the DFS leaf score
    Σ_k w[k]·(0.99·Stat_max[k] + 0.01·Stat_min[k])
as a single INTEGER linear expression (CP-SAT needs integer coefficients). The
`/100` eff rescale and the `0.99/0.01` blend are folded into the coefficients;
float weights are scaled to integers by a factor `W` derived from the model so
the objective stays inside int64.

Isolated here so a later composite/rounding-aware scoring can replace it without
touching the constraint model in model.py. See milp/README.md §5/§7.
"""

import numpy as np

# Keep the worst-case objective magnitude comfortably under int64 (9.22e18).
_INT64_TARGET = 8.0e18


def compute_weight_scale(weights_proj, big):
    """
    Pick the integer weight scale W so that the maximal possible objective
    magnitude stays < _INT64_TARGET. Worst case per stat is
    |wi|·100·(99·BIG + BIG) ≈ |wi|·1e4·BIG (the durability term carries the extra
    ×100), so bound Σ_k |w[k]|·W·1e4·BIG < target.
    """
    aw = np.abs(np.asarray(weights_proj, dtype=np.float64))
    per_unit = float(aw.sum()) * 1.0e4 * float(big) + 1.0
    w = int(_INT64_TARGET / per_unit)
    return w if w >= 1 else 1


def add_objective(mm):
    """
    Build and set the maximisation objective on the model held by `mm`
    (a MilpModel from model.py). Stores `mm.weight_scale` and `mm.int_weights`.

    For eff-scaled stats:  wi[k]·(99·A_max[k] + A_min[k])
    For durability/dura :  wi[k]·100·(99·Stat_max + Stat_min)   (already real units)
    """
    q = mm.query
    W = compute_weight_scale(q.weights_proj, mm.big)
    mm.weight_scale = W

    int_weights = {}
    terms = []
    for k in range(q.stat_count):
        wi = int(round(float(q.weights_proj[k]) * W))
        int_weights[k] = wi
        if wi == 0:
            continue
        if k == mm.dura_idx:
            terms.append(wi * 100 * (99 * mm.dura_max + mm.dura_min))
        else:
            terms.append(wi * (99 * mm.A_max[k] + mm.A_min[k]))

    mm.int_weights = int_weights
    if terms:
        mm.model.Maximize(sum(terms))
    # else: a weightless query (pure feasibility) — leave objective unset.
    return W
