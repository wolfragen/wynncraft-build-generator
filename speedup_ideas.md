# Speedup ideas — parking lot

Ideas discussed but not implemented. Each section captures the *what*, the
*why it's tricky*, and what would need to be true for it to pay off.

---

## Idea 3 — Score-aware upper bound for branch-and-bound

**Status**: parked, requires care for composite stats.

**Today**: `_precompute_bounds` (in `core/search_engine.py`) builds
`future_max_ub[d, s]` = "max contribution to stat `s` from each remaining
slot, summed". The score upper bound at a DFS node is then computed by
combining `current_max[s] + future_max_ub[d, s]` per stat with the user's
weights.

This is **per-stat independent**, so it's loose: the ingredient that
maximizes stat A at slot d is generally *not* the same ingredient that
maximizes stat B at slot d, but the bound assumes both can be achieved
simultaneously.

**Idea**: replace the per-stat bound by a **per-slot best-score-ingredient
bound**. For each remaining slot d, compute the maximum score contribution
of *any single ingredient* placed there (= max over ingredients of
`weights · stat_contrib(ing, eff_d)`). Sum over remaining slots → much
tighter score UB → BB prunes much earlier.

**Why it's hard**:
1. **Composite stats are non-linear in the deps.** The contribution of a
   single ingredient to `mage_meteor` depends on what's *already in the
   build* (skill points, atk speed, weapon dam) and on the *other slots'
   contributions* to the deps (rawToPct, ehp etc. all interact). You can't
   just "score one ingredient in isolation" for composites — the marginal
   value of adding +50 nDamRaw depends on the build's existing nDamRaw,
   sdPct, str count, etc.

2. **Soundness risk**: if the per-slot bound is computed assuming "best
   case for everything else", you over-bound (still sound but loose). If
   you compute "with current state held fixed", you're sound too but only
   for that specific DFS path — and you'd have to recompute per DFS step,
   killing the speedup.

3. **Slot-eff dependency**: each slot has its own eff. The "best
   ingredient for this slot" depends on the eff sign, so bounds are per
   `(eff_sign, slot)` pair, not just per slot. Per-meta-row precompute
   then.

**What it would take**:
- A "linearizable" approximation of composite-stat contribution per
  ingredient. Possibly: linearize around a representative state (e.g., the
  state at the root of the DFS) and accept a slightly over-loose bound.
- Or: skip composite stats in the bound entirely and apply the bound only
  for linear (weighted) stats. Composites still get bounded the old way.
  Less powerful but trivially sound.

**When to revisit**: after #1 (per-eff-sign mask) is in and we have a
clearer profile of where DFS time goes. If linear-stat bounding alone
buys >2x, this is the next big lever.

---

## Idea 5 — Constraint-binding-aware ingredient sort in DFS

**Status**: parked, complementary to #3.

**Today**: `IngredientDB` sorts ingredients by an "approximate score
contribution" heuristic (positive weights × stat_max + negative weights ×
stat_min). The DFS picks ingredients in this order, so high-impact
ingredients are tried first → first valid leaf has high `best_score` →
BB prunes harder thereafter.

**Limitation**: the heuristic ignores **which constraint is binding**. If
the user has `strReq max=20` and the build is currently at strReq=18, the
binding constraint is strReq — the next ingredient pick should
*minimize* strReq, not maximize total score. The current sort would pick
a high-score ingredient that potentially violates the constraint, the
DFS prunes it, retries — wasted work.

**Idea**: at DFS time (per node, not per query), reorder the ingredient
iteration to put **ingredients that respect the binding constraint
first**. Concretely:

1. At the root, identify which constraints are "tight" given the META row
   (e.g., META row already has strReq=15, user max=20 → strReq has 5
   units of headroom).
2. Sort ingredients by `(satisfies_binding_constraint, score)` —
   constraint-respecting ingredients first, then by score.

**Why it's hard**:
1. **The binding constraint changes per DFS node.** What's binding at
   depth 0 might not be binding at depth 3. Recomputing the sort per
   node = expensive. Mitigation: precompute *per stat* a sorted index,
   look up by current binding stat at the node.

2. **Multiple binding constraints**: when 2+ constraints are tight, "sort
   by which" is ambiguous. Could weight by how-tight, but heuristic.

3. **Interaction with #3's BB bound**: if score UB is tight enough,
   constraint-violating branches get pruned early anyway and the sort
   doesn't matter much.

**What it would take**:
- Per-stat-axis sorted index of ingredients (cheap, O(N log N) per stat,
  done once per query → ~150 × 37 = 5K ints).
- DFS code to detect binding constraint at each node (cheap, `current_max[s]
  vs max_constraint[s]`).
- Iterate the right sorted index at each node.

**When to revisit**: alongside #3. They're complementary — #3 makes BB
prune more aggressively, #5 makes the *order of exploration* find good
solutions faster (which feeds #3's bound). Together they could compound.

---

## Implemented (kept here for cross-references)

- **#1 — Per-eff-sign ingredient mask**: precompute, per query, which
  ingredients can ever contribute usefully in a positive-eff slot vs. a
  negative-eff slot. DFS skips masked ingredients per slot. Sound (only
  drops if useless on every active stat). See commit history.
