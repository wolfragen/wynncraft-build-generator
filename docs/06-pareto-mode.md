# 06 — Pareto-Frontier Search Mode

Source: `core/pareto_search.py`, `dsl/{opcodes,builder,evaluator,__init__,test_dsl}.py`,
`main_pareto.py`. Example output: `data/precalc/pareto_example.json`.

An **alternative** to the scalar DFS ([04](04-search-engine.md)). Instead of one
weighted score and one best build, it evaluates each candidate against **K
independent axes** (e.g. DPS, mr, hpBonus) defined as small expressions, and keeps a
**frontier** of non-dominated trade-off builds. A build survives if it reaches at
least a fraction `θ` (threshold) of the current best on **any** axis. It reuses the
same `Query`, `Recipe`, `IngredientDB`, and `META_n` machinery.

---

## The DSL — axes as compiled bytecode

Axes are expressions over projected stats and build context, compiled to flat numba
bytecode so they evaluate inside the DFS.

### `dsl/opcodes.py` — 13 opcodes
- Leaves: `OP_LIT` (constant), `OP_LOAD_STAT` (a projected stat), `OP_LOAD_CTX` (a
  `build_ctx` slot).
- Arithmetic: `OP_ADD`, `OP_MUL` (n-ary, arity in arg0), `OP_SUB`, `OP_DIV`, `OP_NEG`.
- Reducers: `OP_MAX`, `OP_MIN` (n-ary).
- Special: `OP_CLAMP` (3-arg), `OP_SP_HEADLINE`, `OP_SP_ELEMENT` (skillpoint-% lookups).

Each opcode declares per-argument monotonicity in `MONO_TABLE` (`+1` non-decreasing,
`-1` non-increasing, `0` non-monotone). `OP_MUL` is tagged `+1` **assuming
non-negative operands**.

### `dsl/builder.py` — AST → `Program`
You build an `Expr` AST with helpers (`LIT`, `STAT('damPct')`, `CTX(...)`, `ADD`,
`MUL`, …). `compile_program(expr, query)` walks it postorder and emits a `Program`:
three parallel arrays (`op_codes`, `op_args[N,2]`, `consts[K]`) evaluated as RPN on a
float64 stack, plus metadata `max_stack` and `is_monotone`. It resolves stat names
via `query.proj_stats_idx` **at compile time** (raises if the stat isn't active),
interns duplicate constants, and bakes each `LOAD_STAT`'s LB-corner side into
`op_args[i,1]` for fast bounds.

### `dsl/evaluator.py` — bounds evaluation
- `eval_program` — exact value at a point.
- `eval_program_bounds` — `(lb, ub)` over a stat box `[stats_lb, stats_ub]`. For a
  **monotone** program it runs `_eval_corner` twice (per the baked signs) → tight
  bounds in O(N_ops). For a non-monotone program (e.g. dynamic `OP_CLAMP`) it falls
  back to the two extreme corners (conservative but valid).
- SP-lookup opcodes clamp counts to `[0,150]`.

---

## `core/pareto_search.py`

- `pack_axis_programs(programs)` — flattens K `Program`s into the 7 flat arrays the
  kernel needs (offsets/counts per axis + a global const pool).
- `pareto_dfs(...)` (`@njit`) — the recursive DFS. At a leaf it evaluates all K axes
  over the rolling `[current_min,current_max]` box, updates `best_per_axis[k]`, and
  keeps the recipe iff `max_k(axis_ub[k]/best_per_axis[k]) ≥ θ`. Interior nodes prune
  when no axis can possibly reach `θ·best`.
- `search_meta_batch_pareto(...)` — runs `pareto_dfs` over a `META_n` batch, tagging
  each kept recipe with its meta index.
- `run_pareto_search(skill, item_type, recipe_lvl, tier, query, recipe, db, axes,
  threshold, …)` — the orchestrator. Compiles axes, iterates META_0..5, then runs
  two **post-passes**: (1) re-validate survivors against the *final* `best_per_axis`
  (it grew across batches), (2) strict Pareto cull on the UB vectors. Returns a dict
  with `axis_names`, `best_per_axis`, `axes_lb/ub`, `meta_n/index`, `total_searched`.
- `save_frontier(result, db, path)` — reconstructs full 6-slot ingredient ids
  (meta fixed slots + DFS void choices, mapped back via `db.json_ids`) and writes JSON.

---

## `main_pareto.py` — what you edit to run it

1. `build_example_axes()` — return `[(name, Expr), …]`. **Main hook**: write your
   axis expressions here.
2. Set `skill`, `item_type`, `recipe_lvl`, `tier`, and `threshold` (e.g. 0.9 = keep
   within 90% of best on any axis).
3. `ingredient_filter_stats` + `hard_constraints` — every stat referenced by an axis
   **must** be activated here (else `compile_program` raises at compile time).
4. `run_pareto_search(...)` → inspect top-3 per axis → `save_frontier(...)`.

## `pareto_example.json`
```jsonc
{
  "axis_names": ["dps","mr","hpBonus"], "threshold": 0.9,
  "best_per_axis": [74411.02, 84.0, 12082.0], "total_searched": 104597,
  "recipes": [ { "ingredients":[863,894,958,634,793,793],
                 "axes_lb":[42764.95,9.0,10401.0], "axes_ub":[42764.95,9.0,11257.0],
                 "meta_n":5, "meta_index":2784848 }, … ]
}
```

## Gotchas
- `query.dura_proj_idx != -1` is **required** (durability/duration must be active),
  else `run_pareto_search` raises.
- Output buffer overflow sets `out_count[0] = -1` → orchestrator raises asking for a
  bigger `out_buffer_size`. A loose threshold on a big DB can overflow the 2M default.
- `OP_MUL` monotonicity assumes non-negative operands — structure axes so multiplied
  subterms stay ≥ 0, or the bounds get loose (never unsound).
- The threshold gate uses *provisional* `best_per_axis` during the search, so a
  recipe kept mid-search can be dropped in the final re-validation pass.

See session memory `project_pareto_precalc_mode` for the design rationale.
