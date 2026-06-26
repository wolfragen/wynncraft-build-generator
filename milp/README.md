# MILP solver track — design

A second, independent solver for the crafting-optimisation problem, built as an
**exact mixed-integer program** instead of the heuristic branch-and-bound DFS in
[`core/search_engine.py`](../core/search_engine.py). It is the project-specific
adaptation of the general model in
[`../mmorpg_crafting_optimizer.md`](../mmorpg_crafting_optimizer.md) — read that
first for the math; this document is about how it maps onto *this* codebase.

**Status:** design only. No code yet. Lives on branch `mathematical-solve`.

**Intent:** structured for an *eventual replacement* of the DFS, but the first
version deliberately covers **less** than the DFS — see
[§5 Fidelity gaps](#5-fidelity-gaps-vs-the-dfs--craftjs). It is additive: nothing
in the existing DFS/Pareto code is modified.

---

## 1. Why this can live in the same repo cleanly

The MILP needs almost none of the existing engine. The meta/normal split,
`META_n` datasets, `precalc_culling`, the DFS, and the Pareto DSL all exist to
make brute force tractable. The MILP makes them unnecessary: ~200 ingredients ×
6 slots is a small model (~1200 binaries) the solver handles uniformly. So the
new package depends on a **narrow, one-directional slice** of the repo:

```
milp/  ──reuses──▶  data/ingredient_loader      RawIngredient: stats_min/max, pos_mods, skills, id
                    data/ingredient_db.IngredientDB   ingredients → query-projected stat arrays
                    query/ingredient_filter         candidate selection (+ new `include_meta` flag)
                    data/recipe_loader + data/recipe tier-scaled base stats, dura/HP injection
                    query/query.build_query         THE query parser (reused verbatim)
                    utils/hash_generator            ids → crafter URL
                    main_decode.compute_effectiveness   posMods → effectiveness (b-tensor source)
                    main_decode.compute_crafted_stats / score_query   ground-truth oracle

milp/  ──does NOT touch──▶  core/  precalc_*  meta_set_loader  dsl/
```

**Mutualise aggressively.** Everything the MILP needs to go from raw JSON to
query-projected numbers already exists for the DFS — the loader, the
`IngredientDB` projector, the recipe scaler, and the craft.js-faithful
effectiveness/stat/score functions in `main_decode`. The MILP reuses all of them
and adds *only* the optimisation model. The single deliberate behavioural change
in the shared layer is one new flag (`include_meta`, §3) so candidate selection
can return the full ingredient set. That boundary is the whole reason this is a
sub-package and not a separate project: the data layer and the oracle are
**shared, not duplicated**, so the posMods/effectiveness math never forks.

---

## 2. Input contract — identical query, copy/paste parity

> **Hard requirement:** any JSON query accepted by the MILP must run, unchanged,
> on `main.py` by copy/paste. The MILP introduces **no new query syntax.**

This is guaranteed *by construction*, not by convention:

- The MILP calls the **same** [`build_query(...)`](../query/query.py) with the
  **same** `user_json` dict. Same parser → same semantics for `min`, `max`,
  `weight`, `base` / `min_base` / `max_base`, `ingredient_filter`, and
  `_context`. There is no second JSON schema to keep in sync.
- The MILP consumes the resulting `Query` NamedTuple. It never reads the raw
  JSON directly, so it cannot diverge from how `main.py` interprets it.

### Composite rejection

The MILP does not support composite (derived) stats — spell damage, EHP, EHPR,
HPR — for now. `build_query` already flags their presence exactly:

```python
q = build_query(user_json, ...)
if q.comp_count > 0:
    raise UnsupportedQueryError(
        "MILP track does not support composite stats yet "
        f"(query defines {q.comp_count}). Use main.py for this query."
    )
```

`comp_count > 0` ⇔ the user named at least one key in `DERIVED_DEPENDENCIES`
(see [`data/stats.py`](../data/stats.py) and
[`docs/03-query.md`](../docs/03-query.md)). Rejecting here is the *only* place
the two tracks differ on input, and it fails loud and early.

### Query fields the MILP consumes vs. ignores

| `Query` field | MILP use |
|---|---|
| `active_indices`, `stat_count`, `proj_stats_idx` | define the stat set `K` (projected space) |
| `weights_proj` | objective coefficients `w[k]` |
| `min_proj` / `has_min_mask_proj` | hard `Stat[k] >= L[k]` |
| `max_proj` / `has_max_mask_proj` | hard `Stat[k] <= U[k]` |
| `base_min_stats_proj` / `base_max_stats_proj` | additive baseline added into `Stat[k]` (non-req stats) |
| `req_mask_proj`, `req_idx` | requirement stats — see fidelity note on MAX-vs-base |
| `search_for_inversion` | whether `E[p]` lower bound may go negative |
| `item_type`, `skill`, `consumable` | passed to the ingredient loader/filter for the candidate set |
| `comp_*` (all) | **rejected** if present (`comp_count > 0`) |
| `sp_score_*_proj` | **ignored in v1** — SP→% curve is non-linear (fidelity gap §5) |
| `round_offset_proj` | **ignored in v1** — per-stat rounding dropped (fidelity gap §5) |
| `lower_better_proj`, `fast_cull`, `suggested_max_cull`, `full_meta` | DFS-only; irrelevant to the MILP |

---

## 3. The model on this project

Faithful to [`../mmorpg_crafting_optimizer.md`](../mmorpg_crafting_optimizer.md)
with this project's constants plugged in.

### Sets
- `P = {0,1,2,3,4,5}` — the 6 grid slots (DFS numbering, see below).
- `I` — candidate ingredients (see *Candidate set* below).
- `K` — the query's projected active stats (`active_indices`).

### Candidate set — take EVERY ingredient (the `meta_ingredient` flag)

This is a correctness requirement, not an optimisation. The DFS deliberately
*excludes* meta ingredients from the runtime candidate list and handles them via
offline `META_n` enumeration. The MILP has no such split: it must consider
**every legal ingredient in every slot**, or it loses its optimality guarantee.

Three things in the existing selection path would wrongly drop ingredients for
the MILP, all in [`query/ingredient_filter.py`](../query/ingredient_filter.py):

1. **The posMods exclusion** — `if any(x != 0 for x in ing.pos_mods): continue`
   ([line 70](../query/ingredient_filter.py#L70)) drops *every* meta ingredient.
   These are exactly the ingredients with a non-trivial `b[q,i,p]` (§4); omitting
   them removes all effectiveness manipulation from the model.
2. **The `keep` usefulness filter** — keeps an ingredient only if it moves an
   active stat. A posMods-only meta ingredient (no active-stat value, but boosts
   a productive slot's effectiveness) scores as "useless" here and is dropped,
   even though it can raise the objective. Unsafe for the MILP.
3. **The pareto cull** — sound only under the DFS's effectiveness-monotone
   reasoning; it can drop an ingredient that becomes optimal once a neighbour
   boosts its slot. Must be off.

**Plan:** add an `include_meta: bool = False` parameter to
`filter_raw_ingredients`. The MILP calls it with `include_meta=True, cull=False`,
which:
- skips the posMods exclusion (1) and the positive-durability exclusion (the
  other meta marker),
- bypasses the `keep` usefulness filter (2) — returns every ingredient that
  passes only the **legality** gates (`skill`, and the `lvl > recipe.lvl` level
  gate once plumbed; see the TODO at
  [`ingredient_filter.py:36`](../query/ingredient_filter.py#L36)),
- skips the cull (3).

The DFS path is untouched: `include_meta` defaults to `False`, so existing
callers behave identically.

Then the **same** [`IngredientDB(candidates, query)`](../data/ingredient_db.py)
the DFS uses projects `stats_min`/`stats_max` into `active_indices` and gives
`json_ids` — that is the "ingredients → query-dependent stat arrays" tool, reused
verbatim. Its score-sorting and `contrib_*` masks are harmless to the MILP
(ignored). `s[k,i]` (§4) comes straight from `stat_min_matrix` /
`stat_max_matrix`; `json_ids[i]` maps a chosen row back to an id for the b-probe
and the URL.

### Variables
- `y[p,i] ∈ {0,1}` — ingredient `i` in slot `p`. `Σ_i y[p,i] = 1` per slot.
  Reuse is allowed (no `Σ_p y[p,i] ≤ 1`).
- `E[p]` — final effectiveness of slot `p`, in **percent** (integer; base 100).
- `u[p,i] = E[p]·y[p,i]` — exact linearisation (binary `y`), per §10–12 of the
  general doc.
- `Stat[k]` — final crafted value of stat `k`.

### Objective
`maximize Σ_k weights_proj[k] · Stat[k]`. Sign of the weight carries intent
(positive = maximise, negative = minimise), identical to the DFS.

---

## 4. posMods → `b[q,i,p]` (faithful effectiveness)

The general doc's `b[q,i,p]` tensor — "effectiveness bonus to slot `p` when
ingredient `i` sits in slot `q`" — is **fully expressive enough** to encode
Wynncraft's real posMods, because the grid is static:

```
 slot 0 (i=0,j=0) | slot 1 (i=0,j=1)
 slot 2 (i=1,j=0) | slot 3 (i=1,j=1)
 slot 4 (i=2,j=0) | slot 5 (i=2,j=1)
```

Base effectiveness is **100%** for every slot (this project has no per-slot
fixed efficiencies — `base_eff[p] = 100`). For an ingredient `i` with posMods,
placing it in slot `q` shifts a geometry-determined set of target slots:

- `above` / `under` — the whole **column** of `q`, rows above / below.
- `left` / `right` — the single horizontal neighbour cell.
- `touching` — the 4 orthogonal neighbours.
- `notTouching` — every cell that is neither a 4-neighbour nor `q` itself.

`b[q,i,p]` = sum of every posMod of `i` whose geometry, from source slot `q`,
lands on target slot `p` (in percentage points; may be negative). An ingredient
never modifies its own slot: `b[p,i,p] = 0`.

> **Do not re-implement this — derive it from the decoder.**
> [`main_decode.compute_effectiveness`](../main_decode.py) is already the exact
> craft.js mirror of the posMods geometry. The `b[q,i,p]` tensor is *extracted*
> from it by probing one ingredient at a time:
>
> ```python
> # ingredient i alone in slot q, all other slots empty (no posMods):
> probe = [EMPTY]*6
> probe[q] = ingredient_i
> b[q, i, :] = [e - 100 for e in compute_effectiveness(probe)]
> ```
>
> This guarantees the MILP `b` builder and the oracle are literally the **same
> code**, so no fourth divergent copy can drift from `craft.js`.

#### The efficiency model is EXACT, not an approximation

Probing is valid because `compute_effectiveness` is **purely additive across
ingredients**: it starts every slot at 100 and does `eff[k][l] += value` for each
ingredient independently (see [`main_decode.py:157-187`](../main_decode.py#L157-L187)).
Therefore the real per-slot effectiveness is *exactly*

```
E[p] = 100 + Σ_q  b[q, ingredient_at_q, p]
```

which is **identical** to the MILP's linear efficiency equation. So slot
effectiveness is *not* a fidelity gap — the MILP reproduces craft.js
effectiveness exactly. (The only gaps are the per-stat `floor()` and roll-range
collapse — see §5.)

### Effectiveness equation and bounds
```
E[p] = 100 + Σ_{q≠p} Σ_i b[q,i,p]·y[q,i]
E_min[p] = 100 + Σ_{q≠p} min_i b[q,i,p]
E_max[p] = 100 + Σ_{q≠p} max_i b[q,i,p]
```
`min_i`/`max_i` range over all candidate ingredients (including those with no
posMods, i.e. `b=0` — zero may be the best or worst available, per §11).

**Inversion.** `E[p]` may be negative when posMods are strongly negative. The
linearisation keeps `E` continuous with valid signed bounds, so inversion is
handled natively. When `search_for_inversion = False`, clamp `E_min[p]` at 0 so
the solver only explores non-inverted arrangements (matching the DFS default).

### Final stats
```
Stat[k] = base_proj[k] + ( Σ_p Σ_i s[k,i]·u[p,i] ) / 100
```
`s[k,i]` is ingredient `i`'s contribution to stat `k`. `u[p,i]` carries the
percent-scaled effectiveness, so the `/100` rescales once at the end (the general
doc §19 integer-scaling trick). `base_proj[k]` is the query's additive baseline
(`base_min_stats_proj`; req stats handled per §5).

---

## 5. Fidelity gaps vs. the DFS / craft.js

The v1 MILP is exact **for its own linear model**, which is a *simplification* of
the real game rules. Every gap below is intentional and must be documented in the
result output so a user never mistakes a MILP optimum for a craft.js-faithful one.

1. **No composites** — rejected up front (§2). The non-linear ones (spell
   damage, EHP…) are out of scope until a later phase.
2. **No per-stat `floor()` rounding.** craft.js computes `floor(value·e/100)`
   *per slot, per stat*; the MILP sums first and divides once. Results can differ
   by a few units. (`round_offset_proj` is ignored for the same reason.)
3. **No skillpoint→% non-linear scoring.** `sp_score_*_proj` drives a non-linear
   curve for str/dex/int/def/agi in the DFS. The MILP scores SP stats as a plain
   linear `weight·Stat` like any other stat. Queries that lean on SP-cap scoring
   will rank differently.
4. **Roll ranges → point values.** Ingredient stats roll `[min,max]`. The MILP
   needs a scalar `s[k,i]`. v1 decision: use the same objective blend the DFS
   uses (`0.99·max_roll + 0.01·min_roll`) for the objective, and the optimistic
   `max_roll` for feasibility against hard `min`/`max`. *(Verify against the
   exact DFS constraint handling in `core/search_engine.py` at implementation
   time — this is the one modelling choice most likely to need adjustment.)*
5. **Requirement MAX-vs-base.** For `*Req` stats the DFS combines the user's
   `base` with the craft via **MAX**, not sum. v1: model the constraint on the
   craft value alone (`build_query` already validates `base ≤ user max`), which
   matches the DFS for the common case; revisit if a query weights a req
   positively.

Because the input is the identical `Query`, these gaps are differences in the
**objective/feasibility fidelity**, never in what the query *means*. They are
walled behind `milp/objective.py` so a later, composite/rounding-aware version
can slot in without touching the constraint model.

---

## 6. Solver: CP-SAT

OR-Tools **CP-SAT**, for two forward-looking reasons aligned with "eventual
replacement":

1. **Integer-native** — fits the percent / basis-point integer scaling above and
   avoids float precision issues entirely.
2. It is the only free backend that can later express the **non-linear**
   composite stats (variable × variable) via `AddMultiplicationEquality` /
   reification — so we will not have to swap solvers when fidelity grows. A pure
   MILP/CBC backend (PuLP) could do v1 but would be a dead end for composites.

Map solver status → result: `OPTIMAL` (proven best), `FEASIBLE` (time-limited,
not proven), `INFEASIBLE` (no craft satisfies the hard min/max — a valid answer,
report it clearly per general-doc §20), `MODEL_INVALID` / `UNKNOWN` (surface the
error). Only `OPTIMAL` may be reported as "mathematically optimal".

---

## 7. Planned package layout

New code is small and additive; the heavy lifting is reused.

```
milp/
  README.md         this document
  __init__.py
  efficiency.py     b[q,i,p] by probing main_decode.compute_effectiveness (no new geometry)
  model.py          build y / E / u / Stat, assignment, linearisation, min/max
  objective.py      SEAM: linear weighted sum today; composite/rounding-aware later
  solve.py          CP-SAT wrapper → status + assignment
  result.py         decode y[p,i]=1 → ids → URL + decoder-verified report
../main_milp.py     entry point, mirrors main.py: load → build_query → reject-composites
                    → select candidates (include_meta=True) → IngredientDB → solve → URL
../requirements-milp.txt   isolate the ortools dependency
```

Reused unchanged: `data/ingredient_loader`, `data/ingredient_db.IngredientDB`,
`data/recipe_loader` + `data/recipe`, `query/query.build_query`,
`utils/hash_generator`, `main_decode.{compute_effectiveness, compute_crafted_stats,
score_query}`. The only shared-layer edit is the `include_meta` flag on
`query/ingredient_filter.filter_raw_ingredients` (§3).

> **Import caveat.** `main_decode.py` runs a block of module-level demo code
> (weapon-weight constants, `if __name__` aside). Importing its functions
> executes that block. Before the MILP imports it, factor the three pure
> functions (`compute_effectiveness`, `compute_crafted_stats`, `score_query`)
> into an import-safe module (e.g. `craft_core.py`) that both `main_decode` and
> `milp` import — this also shrinks the "three places must agree" surface the
> architecture doc warns about.

---

## 8. Validation — reuse the decoder as the oracle

No new scoring code. The decoder's pure functions ARE the reference:

- **Effectiveness is exact by construction** (§4) — the b-tensor is probed from
  `compute_effectiveness`, so `E[p]` matches craft.js with no drift to check.
- **True score of the MILP's pick.** Map the chosen `y[p,i]=1` to 6 ids, then
  call `compute_crafted_stats(...)` + `score_query(crafted, user_json)` directly
  (no URL round-trip needed) to get the craft.js-faithful stats, the
  VALID/INVALID verdict, and the reference score. Because of the §5 gaps (floor,
  roll-collapse) expect small numeric drift between the MILP's internal objective
  and this reference — assert the pick is a *legal* craft and report both numbers.
- **Upper bound on the DFS.** On a composite-free linear query the MILP `OPTIMAL`
  objective is a provable ceiling. Run both tracks: if the DFS exceeds it, a
  search bug exists; if it falls short, the §5 gaps or a DFS issue explain it.
  This mutual check is the main payoff of keeping both solvers in one repo.

---

## 9. To verify at implementation time

- Exact DFS handling of roll ranges in hard `min`/`max` constraints
  ([`core/search_engine.py`](../core/search_engine.py)) — pins down the §5.4
  scalar-collapse decision.
- Whether any candidate ingredient can modify its own slot in craft.js (assumed
  `b[p,i,p]=0`; the probe in §4 will show it directly).
- craft.js effectiveness clamping under deep inversion vs. the `E_min` clamp.
- Recipe-base injection (durability scaling, armor HP→`hpBonus`) into `base_proj`
  — mirror exactly what `compute_crafted_stats` / `data/recipe.py` do so the MILP
  and the oracle agree on the baseline.
