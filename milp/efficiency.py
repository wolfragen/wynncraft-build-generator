"""
milp/efficiency.py

Build the effectiveness tensor b[q,i,p] = "percentage-point change to slot p when
candidate ingredient i is placed in slot q", by PROBING the craft.js-faithful
`craft_core.compute_effectiveness` one ingredient at a time. This guarantees the
MILP's effectiveness math is the SAME code as the decoder/oracle — no second
implementation to drift.

Because `compute_effectiveness` is purely additive across ingredients (it starts
every slot at 100 and does `eff[k][l] += value` per ingredient independently), the
real per-slot effectiveness of any full build is EXACTLY

    E[p] = 100 + Σ_q b[q, ingredient_at_q, p]

which is the MILP's linear efficiency equation. So this is exact, not an
approximation. See milp/README.md §4.
"""

import numpy as np

from craft_core import compute_effectiveness


# A slot with no posMods. `compute_effectiveness` only reads `.get("posMods")`,
# so an empty dict contributes exactly zero everywhere. Must be a dict (never
# None — `.get` would crash).
EMPTY = {}


def build_b_tensor(json_ids, ing_by_id, search_for_inversion=True):
    """
    Args:
        json_ids: sequence of N candidate ingredient JSON ids, in the row order
            the model will use (typically `IngredientDB.json_ids`, post-sort).
        ing_by_id: dict id -> raw JSON ingredient entry (from
            craft_core._index_ingredients), the SAME entries the oracle reads.
        search_for_inversion: if False, clamp E_min at 0 (restrict the model to
            non-inverted arrangements); if True, keep the true signed lower bound.

    Returns:
        b      : np.int32 [6, N, 6]  — b[q,i,p] in percentage points; b[q,i,q]==0.
        E_min  : np.int64 [6]        — per-slot min achievable effectiveness (%).
        E_max  : np.int64 [6]        — per-slot max achievable effectiveness (%).

    E_min/E_max range over ALL candidates, including those with no posMods
    (b==0), because zero may itself be the extreme contribution from a source
    slot (mmorpg_crafting_optimizer.md §11).
    """
    n = len(json_ids)
    b = np.zeros((6, n, 6), dtype=np.int32)

    for i, jid in enumerate(json_ids):
        entry = ing_by_id[int(jid)]
        for q in range(6):
            probe = [EMPTY] * 6
            probe[q] = entry
            eff = compute_effectiveness(probe)   # 6 ints, base 100
            for p in range(6):
                b[q, i, p] = eff[p] - 100

    # An ingredient never modifies its own slot (craft.js geometry excludes the
    # source cell). Assert it empirically — a failure means a data/JS surprise
    # the model's b[p,i,p]=0 assumption relies on. (README §9 open question.)
    for q in range(6):
        if not np.all(b[q, :, q] == 0):
            raise AssertionError(
                f"b[{q},i,{q}] != 0 for some i: an ingredient modifies its own slot."
            )

    # Per-(source q, target p) extrema over the candidate axis. Each source slot
    # contributes exactly one selected b term, so summing the per-source extrema
    # gives exact reachable bounds on E[p].
    per_qp_min = b.min(axis=1)   # [6, 6]
    per_qp_max = b.max(axis=1)   # [6, 6]

    E_min = np.full(6, 100, dtype=np.int64)
    E_max = np.full(6, 100, dtype=np.int64)
    for p in range(6):
        for q in range(6):
            if q == p:
                continue
            E_min[p] += int(per_qp_min[q, p])
            E_max[p] += int(per_qp_max[q, p])

    if not search_for_inversion:
        E_min = np.maximum(E_min, 0)

    return b, E_min, E_max
