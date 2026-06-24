# 01 — Architecture & Data Flow

## What problem this solves

Wynncraft lets players **craft** items by placing up to **6 ingredients** into a
2×3 grid on top of a **recipe** (the base item: a chestplate, a ring, a potion…).
The crafted item's stats are the recipe's base stats plus each ingredient's stats,
**scaled by that ingredient's slot effectiveness**. Effectiveness is not fixed: an
ingredient's `posMods` raise or lower the effectiveness of neighbouring slots.

The generator answers: *given a recipe and a weighted stat query, which 6-ingredient
arrangement produces the highest-scoring legal crafted item?* It then prints a
[wynnbuilder crafter URL](https://wynnbuilder-beta.github.io/crafter/) for that build.

The search space is huge (≈150 candidate ingredients to the 6th power × slot
arrangements), so the optimiser uses two ideas in combination:

1. **Split ingredients into "meta" and "normal"** and precompute the hard part offline.
2. **Branch-and-bound DFS** at runtime to fill the rest, with numba-compiled kernels.

## The grid, effectiveness, and posMods

Slots are numbered `0..5`, laid out as a 3-row × 2-column grid:

```
 slot 0 (i=0,j=0) | slot 1 (i=0,j=1)
 slot 2 (i=1,j=0) | slot 3 (i=1,j=1)
 slot 4 (i=2,j=0) | slot 5 (i=2,j=1)
```

Each slot starts at **100% effectiveness**. Every ingredient may carry `posMods`
that shift the effectiveness of other slots:

- `above` / `under` — affect the **whole column**, the rows above / below.
- `left` / `right` — affect the **single** horizontal neighbour cell.
- `touching` — the 4 orthogonal neighbours.
- `notTouching` — every cell that is *not* a 4-neighbour (and not itself).

A stat rolled `[min,max]` on an ingredient at slot `s` with effectiveness `e`
contributes `floor(value * e / 100)` to the crafted item (requirements use a
`Math.round` bias instead — see [08](08-glossary-and-gotchas.md)). When `e` is
**negative** (possible under "inversion"), the contribution sign flips and the
min/max bounds swap. This whole computation mirrors `craft.js` and is
re-implemented in three places that must agree: `precalc_fast._grid_kernel`,
`core/search_engine` (the DFS contribution step), and `main_decode.compute_*`
(the verification tool).

> Ground truth for posMods semantics: `wynnbuilder.github.io-master/crafter/craft.js`
> (~lines 425–432). See session memory `reference_posmods_semantics`.

## Meta vs normal ingredients — the key idea

- **Meta ingredients** have `posMods` (or `ingredEff`, charges, or *positive*
  durability). They change other slots' effectiveness, so *their arrangement
  matters* and the per-slot effectiveness grid can only be known once you fix
  where they all go. There are relatively few of them, so all their arrangements
  are enumerated **offline**.
- **Normal ingredients** have no posMods. They never change anyone's
  effectiveness, so a normal ingredient's contribution depends only on *which
  slot's effectiveness it lands in* — not on the other ingredients. They are
  searched **at runtime** to fill the leftover ("void") slots.

This is what makes the problem tractable: instead of enumerating all 6-ingredient
combinations, the offline pass enumerates only meta arrangements, and the runtime
DFS greedily fills voids with normal ingredients.

### META_n datasets

A **`META_n`** dataset contains arrangements with exactly `n` fixed meta
ingredients and `6 − n` **void slots**. `n` ranges `1..5`. Each row stores the
fixed ingredient ids, the resulting 6-slot effectiveness vector, and the
pre-summed stat ranges of the fixed part. `META_0` is the synthetic "no meta
ingredients, all 6 slots void" row, built in memory.

> **Important consequence (by design):** a craft whose **all six** slots are meta
> ingredients has *no* void slot and is unreachable by the normal pipeline
> (`META_n` stops at `n=5`). The optional `full_meta` pass recovers some of these.
> See caveat #1 in [`../README.md`](../README.md).

## End-to-end flow (`main.py`)

```
load_ingredients(ingreds_compress.json)            data/ingredient_loader.py
        │   → list[RawIngredient]  (dense int16 stat vectors)
        ▼
build_query(user_json, item_type, skill, …)        query/query.py
        │   → Query  (full + projected dense arrays, composites, masks)
        ▼
find_recipe(...) → build_recipe(..., tier)         data/recipe_loader.py + data/recipe.py
        │   → Recipe (tier-scaled base stats injected into projected space)
        ▼
filter_raw_ingredients(ingredients, query, recipe) query/ingredient_filter.py
        │   → keeps only "normal" ingredients that can matter, then Pareto-culls
        ▼
IngredientDB(filtered, query)                      data/ingredient_db.py
        │   → compact, score-sorted matrices for the DFS
        ▼
search_pipelined(skill, query, recipe, db, …)      core/search_engine.py
        │   ├─ background thread: load_meta_sets + per-tier cull   data/meta_set_loader.py
        │   └─ foreground: branch-and-bound DFS fills void slots, scores leaves
        │   (optional) _search_full_meta extends META_5 rows       core/search_engine.py
        ▼
best 6 ingredient ids → generate_crafter_url(...)  utils/hash_generator.py
```

`warm_numba()` (`core/warmup.py`) runs first to JIT-compile every kernel so the
first real search isn't penalised.

## Scoring objective

A leaf (complete build) is scored as:

```
score = Σ_stats   weight_s · (0.99 · max_roll_s + 0.01 · min_roll_s)
      + Σ_composites weight_c · (0.99 · cmax_c   + 0.01 · cmin_c)
```

The `0.99/0.01` blend is a "expect near-max rolls" heuristic (caveat #7). Hard
constraints (`min`/`max` per stat) reject a leaf entirely rather than penalising
it. **Composites** are derived stats (spell damage, EHP, EHPR, HPR) computed from
several base stats — see [03](03-query.md) and [04](04-search-engine.md).

## Two search modes

- **Scalar DFS** (`core/search_engine.py`, driven by `main.py`) — one weighted
  objective, returns the single best build. This is the primary mode.
- **Pareto frontier** (`core/pareto_search.py`, driven by `main_pareto.py`) —
  K independent axes defined via a small DSL, returns a frontier of
  non-dominated trade-off builds. See [06](06-pareto-mode.md).

Both reuse the same `Query`, `Recipe`, `IngredientDB`, and `META_n` machinery.

## Verifying a result

`main_decode.py` decodes any crafter URL, recomputes the full crafted stats from
first principles (mirroring `craft.js`), and—given a query—reports the weighted
score plus a VALID/INVALID verdict. It is the **ground-truth oracle** for "is our
search output actually correct/optimal vs. someone else's build?" See
[07](07-encoding-and-tools.md).
