# 04 — The Search Engine

Source: `core/search_engine.py` (largest, most important file) and `core/warmup.py`.

A numba-compiled **branch-and-bound depth-first search** that fills the void slots
of each `META_n` row with normal ingredients from the `IngredientDB`, maximising
the weighted score subject to hard constraints. "void_count = k" is the number of
slots to fill (`k = 6 − n`).

---

## Entry points

### `search_pipelined(skill, query, recipe, db, max_cull=5, culling=True, semi_cull_budget_s=0.0, …)`
The one `main.py` calls. Overlaps **meta-set loading + culling** (background
producer thread) with **searching** (foreground consumer) via a bounded queue, so
wall-clock ≈ `max(load, search)` instead of their sum. Prints a Load/Search/Wall
breakdown.

Per-`META_n` cull policy:
- `n ≤ max_cull` → full Pareto cull;
- `max_cull < n ≤ semi_cull_max_n` with a budget → time-boxed "semi-cull" (keeps a
  valid Pareto *superset* if the budget runs out);
- larger `n` → pass through.

`max_cull` is supplied as `query.suggested_max_cull` (= 3). See session memory
`project_semi_cull_tradeoff` for why semi-cull only helps on ~1–2M-row metas.

If `query.full_meta` is set, after the normal search it runs `_search_full_meta`
and keeps whichever build scores higher.

### `search(all_meta_sets, db, query) -> int32[6] | None`
Processes batches **largest-M-first** (shallowest DFS) so an early tight
`best_score` accelerates pruning later. Dispatches each batch by `void_count`:
`k=1` → `_search_meta_batch_k1`, `k=2` → `_search_meta_batch_k2`, `k≥3` →
`search_meta_batch_v2`. Returns the best 6 ingredient ids.

### `_search_full_meta(skill, query, recipe, base_path, chunk_rows=4000)`
Recovers all-6-meta builds: for each surviving `META_5` row, drops **every** meta
ingredient into its single void slot, recomputes the full grid via
`precalc_fast._grid_kernel`, and scores the complete 6-meta build. Chunked for
memory. Sound (same projection + scoring as the normal path) but expensive, hence
opt-in. Addresses caveat #1.

---

## The DFS (`dfs`, plus inlined `k1`/`k2` fast paths)

Recursion over depth `0..k`:

**Leaf (`depth == k`)** — build complete:
1. For each stat: derive final `[min,max]`, reject if it violates a hard constraint,
   apply the skillpoint cap clamp for SP stats, add `weight·(0.99·max + 0.01·min)`.
2. For each composite: evaluate the formula on the finalised ranges, check its
   constraints, add its weighted contribution.
3. Update `best_score` / `best_solution` if it wins.

**Interior (`depth < k`)** — for each remaining ingredient `i`:
1. **Useful-mask prune** — skip if `useful_pos_eff[i]` / `useful_neg_eff[i]` says
   `i` cannot help any active stat/composite at this slot's eff sign.
2. **Dura prune** — skip if durability can't reach its min even with best remaining slots.
3. **Apply `i`**: `contribution = (db_stat · eff + round_offset) // 100`, **swapping
   min/max when `eff < 0`** so `current_max ≥ current_min` always holds (true
   reachable extrema).
4. **Fused feasibility + score upper-bound check** (the heart of the pruning):
   - Feasibility: if `current_max[s] + future_max_ub[next_depth][s] < min_vals[s]`,
     the subtree can never satisfy stat `s` → skip.
   - Score UB: combine current state with precomputed suffix bounds; if
     `ub_score ≤ best_score`, prune the whole subtree.
5. **Permutation kill** — when the next slot has the same eff, start its loop at `i`
   (avoid re-counting permutations of identical-eff slots).
6. Recurse, then **undo** `i` (backtrack).

### Eff-sign-aware contribution
Slot eff can be negative under inversion. For `eff ≥ 0`, larger roll → larger
contribution. For `eff < 0`, larger roll → more-negative contribution, so the min
and max bounds **swap**. Every place that scales a stat by eff must honour this
(DFS step, `k1`/`k2`, `_precompute_bounds`, and the offline cache correction in
`meta_set_loader`).

---

## Pruning machinery

### `_compute_useful_masks` → `useful_pos_eff[N]`, `useful_neg_eff[N]`
Per ingredient × eff-sign: can this ingredient affect **any** active constraint,
weight, or composite at that sign? Composite directions are induced from deps
assuming monotone-increasing formulas; `mul_div_100` is non-monotone, so its deps
are marked useful in **both** directions (conservative). Drops an ingredient only
if it's strictly useless at that sign for *every* active stat.

> ⚠️ Interacts with caveat #3 in [`../README.md`](../README.md): because slots can't
> be left empty, the mask can prune the *least-harmful* fill and force a worse one.

### `_precompute_bounds` → suffix-sum tensors
For depth `d`, `future_*_ub/lb[d][s]` is the best/worst additional contribution to
stat `s`'s min/max from slots `d..k-1`, eff-sign-aware. Row `k` (empty suffix) is
zero. Feeds both the feasibility and score-UB checks.

### `_compute_slot_best_per_eff` → tighter score UB
The naive per-stat-independent UB lets *different* ingredients maximise *different*
stats in the same slot (impossible — one ingredient per slot). This precomputes,
per distinct eff, the max **total linear score** a single ingredient can add, then
sums over remaining slots (`score_suffix_ub`). Excludes SP-cap stats and composites
(non-linear). This is "Idea 3 (partial)" from `speedup_ideas.md`.

---

## Composite evaluation & bounding

At a **leaf**, composites are evaluated exactly via helpers:
`_raw_to_pct`, `_product_bounds_div100`, `_ehp_bounds`, `_ehpr_bounds`, and the
spell corner evaluators `_eval_spell_corner_spell` (36 deps) /
`_eval_spell_corner_melee` (32 deps).

During **B&B**, each dep `d` is bounded by the rectangle
`[current[d] + future_min_lb, current[d] + future_max_ub]`, and the formula's
admissible min/max is computed over that rectangle (4-corner for products,
sign-aware for raw→pct, bilinear for ehp/ehpr, corner-eval for spells). The
directional pick uses the UB for positive weight, LB for negative.

> **Monotonicity assumption (caveat #6):** spell formulas are treated as monotone in
> each dep, so corner evaluation bounds them. Holds for sane (single-sign) rolls; a
> dep range that straddles zero could yield an inadmissible bound. Not guarded.

### Skillpoint-cap-aware scoring
SP bonuses cap at 150 total. The player must allocate `max(all xReqs)` to meet
requirements, leaving `150 − ctx_base − max_req` for free bonus. The leaf clamps SP
ranges by that exact cap; the B&B uses a looser per-state bound (each future slot's
cap considered independently — sound, not tight).

---

## Parallelism & specialised paths

- `_search_meta_batch_k1` — single void slot, no recursion, `prange` over meta-sets.
- `_search_meta_batch_k2` — two slots, both inlined, B&B kept alive via a slot-1 UB.
- `search_meta_batch_v2` (`k≥3`) — parallel over `(meta_set m × first ingredient i0)`,
  then serial DFS for the rest. Exposes `M×N` parallelism (v1 stranded cores on
  low-M/high-k batches like META_6). Going further (parallel `i1`) was measured to
  hurt by weakening intra-worker B&B.

Cross-thread `shared_best` reads/writes are racy-but-benign: worst case is weaker
pruning, never a lost solution (each worker keeps its own `best_score_ref`).

---

## numba specifics & `warmup.py`

- Most kernels are `@njit(cache=True)`; parallel ones `@njit(parallel=True, cache=True)`.
- The recursive `dfs` needs an **explicit eager signature** (`_DFS_SIG`) for
  `cache=True` — numba can't otherwise resolve the self-reference. See session
  memory `project_numba_cache_recursion`: self-recursive `@njit(cache=True)`
  segfaults without an eager signature.
- `warm_numba()` (called at startup) forces compilation of every kernel and **every
  composite branch** (all 13 spells, all fixed formulas, all UB paths, k1/k2/v2), so
  the first real search pays nothing.

---

## Soundness caveats (cross-reference)

The search is **not guaranteed globally optimal**. The authoritative list lives in
[`../README.md`](../README.md) (caveats #1–#7). The most load-bearing here:
- #1 all-6-meta builds unreachable without `full_meta`;
- #3 void slots are force-filled + useful-mask can force a worse-than-empty fill;
- #5 runtime meta-row cull keeps a single representative per stat (lossy on the
  `0.01·min` term and constraints) — mitigated by only culling META_1..3.
