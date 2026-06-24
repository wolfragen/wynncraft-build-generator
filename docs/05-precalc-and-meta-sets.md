# 05 — Precalc Pipeline & Meta-Sets

Source: `precalc_fast.py`, `precalc_culling.py`, `precalc_culling_iter.py`,
`data/meta_set_loader.py`, `query/ingredient_filter.py`.

This is the **offline half** of the optimiser. It enumerates every arrangement of
meta ingredients, Pareto-culls them, and writes `META_n.json` files that the
runtime search loads and culls again per query.

See [01](01-architecture.md) for the meta/normal split and the `META_n` definition.

---

## Why split, recap

Meta ingredients (those with `posMods` / `ingredEff` / charges / positive
durability) change neighbours' effectiveness, so their **arrangement** determines
the 6-slot effectiveness grid. There are few of them, so all arrangements are
enumerated offline. Normal ingredients don't change effectiveness, so they're left
as **void slots** filled at runtime. A `META_n` row = `n` fixed meta ingredients +
`6−n` void slots, storing the fixed ids, the resulting 6-slot eff vector, and the
pre-summed stat ranges of the fixed part.

---

## `precalc_fast.py` — enumerate + pack

`run_precalculation_fast(profession, pre_filled=5, …)` runs `_run_one` for `n=1..5`.

Pipeline per profession/`n`:
1. **`_filter_and_pack`** — select the profession's meta ingredients, build dense
   numpy arrays for the JIT kernels. Stat order is preserved: ids (JSON order) →
   reqs (fixed) → durability → charges. Marks `is_req_global[k]` for the 5 reqs.
2. Enumerate, per multiset of `n` ingredients, every unique permutation × every
   `C(6,n)` slot placement.
3. **`_grid_kernel`** (`@njit`) — for one placement: start every slot at 100,
   add `ingredEff`, propagate `posMods` to neighbours, then accumulate stats with
   eff scaling (`value * e // 100` for scalable stats; dura/charges raw). Stores the
   **raw** convention `db_min*e → "min"`, `db_max*e → "max"` regardless of eff sign.
4. Durability feasibility check, then **`_pack_for_pareto`** + **`_local_cull_jit`**
   Pareto-cull within the multiset.
5. Stream survivors to `data/precalc/<...>/META_n.json` (one object per line).

> **Baked-in assumption — "requirements are lower-better" (caveat #4).**
> `is_req_global` makes `_compare_recipes_jit` treat every `*Req` as strictly
> lower-better when deciding dominance. So high-requirement arrangements are
> **culled offline**. A query that *wants* a requirement high can never see them.
> See session memory `project_cull_direction_fix` (the *runtime* cull was fixed to
> use `lower_better_proj`, but these precalc files still bake the legacy rule).

### Negative-eff "raw" convention and the later swap
`_grid_kernel` writes `db_min*e → min`, `db_max*e → max` even when `e < 0` — which
is *wrong* for negative eff (`db_max*e` is the more negative). The **cache builder**
in `meta_set_loader` applies an eff-sign-aware swap to fix it (see below). The JSON
keeps the raw convention; the `.cache.npz` stores corrected true bounds.

---

## `precalc_culling.py` / `precalc_culling_iter.py` — global cull

`precalc_fast` culls *locally* (within each multiset). These do a **global** Pareto
cull across all multisets — more aggressive, fewer survivors — and write to
`data/precalc/generic_cull/<SKILL>/META_n.json`.

- `precalc_culling.cull_recipes(...)` — loads the whole file into one
  `(rows, num_effs + num_stats)` matrix, sorts by a strength heuristic, runs a
  block-parallel Pareto filter. Stat categories: invertible (higher-better),
  req (lower-better), dura (strictly higher-better, non-invertible).
- `precalc_culling_iter.py` — **streaming** version for files too big for RAM
  (ARMOURING/META_5 ≈ 7 GB). `cull_recipes_streaming_v2` is the optimised path
  (intra-batch cull, then batch-vs-survivors scan, with resumable checkpoints).
  Identical survivor set to the in-RAM version.

> Validation note (session memory `feedback_cull_comparison` /
> `project_final_validation`): validate a new cull by **crafter-output diff on an
> identical query vs single-thread**, *not* by file-level diff and *not* against the
> pre-bugfix archive `generic_cull_archive/2026-02-24_pre-bugfix/`.

### `META_n.json` schema (one object per line)
```jsonc
{
  "ings":  [-1, -1, 792, 895, -1, -1],   // JSON ids; -1 = void slot
  "eff":   [115, 85, 100, 255, 115, 85], // per-slot effectiveness %
  "stats": { "hpBonus": {"min":370,"max":508}, "defReq": {"min":28,"max":28},
             "durability": {"min":-112,"max":-112}, "...": {} }
}
```

---

## `data/meta_set_loader.py` — runtime loading + per-query cull

`load_meta_sets(skill, query, recipe, culling, max_cull, base_path)` returns a list
`[META_0 … META_5]` of `MetaBatch` NamedTuples. For each:

1. **`_load_cached_arrays`** — reads a fresh `META_n.cache.npz` if present and
   `version == _CACHE_VERSION`; else parses the JSON once via **`_build_cache`** and
   writes the cache. `_CACHE_VERSION` is bumped when the on-disk format changes
   (currently 3) so stale caches are ignored. The cache stores
   `ings, eff, stat_names, stat_min, stat_max`.
2. **Inversion correction** (`_apply_inversion_correction` → `_correct_inversion_kernel`)
   — applies the eff-sign-aware swap the precalc deferred: for any non-void slot with
   `eff < 0`, swaps the per-slot min/max contribution so the cache holds the **true
   reachable** bounds the search engine expects. (This is what `_CACHE_VERSION` v2/v3
   are about.)
3. **`_refine_batch`** — projects stats into the query's active space, applies the
   recipe shift, extracts the void-slot effs, runs the numba Pareto cull
   (`numba_cull`) when `culling and n ≤ max_cull`, then sorts each row's void effs
   descending.

`MetaBatch` fields: `ings_matrix (M,6)`, `void_count`, `void_eff_matrix (M,void)`,
`void_slots_matrix (M,void)`, `base_min_matrix (M,K)`, `base_max_matrix (M,K)`.

### The runtime cull (`numba_cull` and friends)
- `_build_cull_matrix` collapses each stat's `[min,max]` to **one** representative
  (the `lower_better_proj`-preferred bound) and lays out `[sorted void effs | stats]`.
- `compare_vectors` decides dominance; `pareto_filter` (sequential, reference) and
  `pareto_filter_block` (block-parallel, with optional `time_budget_s` semi-cull)
  apply it. The block path sorts rows by "strength" so strong dominators fill the
  survivor set first.

> Single-representative collapse is **lossy** (caveat #5): it can drop a row worse on
> the chosen bound but better on the other, which still matters for constraints and
> the `0.01·min` score term. Mitigated by only culling META_1..3 (`suggested_max_cull=3`).

Performance notes (session memory `project_pareto_cull_perf`): in numba `x in arr`
is a linear scan; sort-by-strength before culling is a big win; parallel
intra-block gives no win.

---

## `query/ingredient_filter.py` — selecting *normal* ingredients

`filter_raw_ingredients(ingredients_raw, query, recipe, cull=True)` keeps the normal
ingredients worth searching. With `cull=True` (default, used by `pareto_search` and any
single-DB caller) it then Pareto-culls them. With `cull=False` it returns the
keep-filtered list **uncut** — the scalar search (`main.py`) passes this and then builds
the **dual database** (below).

Filter steps per ingredient:
- **Skill filter** — must support the query's skill.
- **posMods filter** — `if any(x != 0 for x in ing.pos_mods): continue`. **Any
  ingredient with a non-zero posMod is dropped here** — those are *meta* ingredients,
  handled entirely by the offline `META_n` pipeline, not the runtime DB.
- **Durability/duration sign filter** — drop positive-durability (or
  positive-duration / non-zero-charges for consumables) ingredients.
- **Per-stat keep test** — keep if the ingredient can usefully move any
  query-relevant stat in the needed direction (including inversion branches when
  `search_for_inversion`).

The legacy `pareto_cull_ingredients` runs either `_pareto_cull_exact` (range-aware,
stores both bounds) or `_pareto_cull_fast` (single-representative, gated by
`query.fast_cull`). **Both are unsound under inversion** — see the dual-DB note next.

### The dual database (per-effectiveness-sign split) — the sound cull

The `_pareto_cull_exact` `search_inv=True` branch uses **range-containment** dominance
(A dominates B iff A's `[min,max]` contains B's). That is wrong against the leaf's
`0.99·max + 0.01·min` blend: a wide range like `xpb [3,8]` "contains" and so wrongly
drops a tight high range `xpb [7,8]`, even though `[7,8]` is the *better* filler for
negative-eff slots (its higher floor gives a more-negative blended contribution). An
ingredient's per-stat value is a **V-shape in slot effectiveness** (one ray for `eff>0`,
one for `eff<0`), so two ingredients can each win at a different sign — genuinely
incomparable on one list.

Fix: **split candidates into two databases by eff sign.** Within a fixed sign the value
is *monotone*, so the plain non-inversion directional dominance is sound.

- `split_and_cull_by_sign(filtered, query, recipe)` → `(pos_list, neg_list)`:
  - **Membership** via `_useful_masks_split` (same logic as
    `search_engine._compute_useful_masks`, but skips durability): an ingredient joins a
    sign's list iff it can help some active stat/composite at that eff sign. For a
    negative-weighted stat, positive-roll ingredients land only in the *negative*-eff
    DB, negative-roll ones only in the *positive*-eff DB (and vice-versa for positive
    weight). An ingredient useful at both signs joins both lists.
  - **Per-sign cull** via `_cull_signed` → `pareto_filter_ingredients(..., search_inv=False)`
    with direction = `lower_better_proj` for the pos DB and **flipped** for the neg DB
    (because `eff<0` inverts roll→contribution). Durability is never eff-scaled and stays
    higher-better (the cull special-cases `dura_proj_idx`).
- `data/ingredient_db.build_dual_db(pos_list, neg_list, query)` concatenates the two
  score-sorted regions into one DB: rows `[0, pos_count)` = positive-eff DB,
  `[pos_count, count)` = negative-eff DB. The search kernels read `db.pos_count` and make
  each void slot iterate only its sign's region (`dfs`/`k1`/`k2`/`v2`). `json_ids` is
  concatenated so void-fill indices map straight back with no per-sign bookkeeping.

Result: smaller per-slot candidate sets (faster) **and** the sound cull recovers builds
the old range-containment dropped (e.g. the `xpb` minimize query: 22432 → 32174).

> This `posMods` drop is exactly why a stats-less pure-effectiveness ingredient like
> *Borange Fluff* (only `posMods: {touching:15, notTouching:15}`) never appears in
> the runtime DB — it's a meta ingredient and lives only in the `META_n` precalc.
> If the offline enumeration/cull dropped the arrangement that uses it, the runtime
> search cannot rediscover it. (This was the lead in the URL-comparison
> investigation that kicked off this documentation.)

---

## Regenerating the precalc

```bash
python precalc_fast.py        # enumerate raw meta-set arrangements (data/precalc/full or similar)
python precalc_culling.py     # global Pareto cull → data/precalc/generic_cull/<SKILL>/
# or precalc_culling_iter.py for the streaming (low-RAM) path on the big files
```
Delete stale `*.cache.npz` siblings if you change the JSON or bump `_CACHE_VERSION`.
