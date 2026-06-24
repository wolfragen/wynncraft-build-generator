# 03 — The Query

Source: `query/query.py`.

`build_query(...)` turns the user's stat-constraint dict into a `Query` NamedTuple
of dense numpy arrays. The `Query` is the single source of truth for *what the
search optimises and what it rejects*. It keeps two parallel views:

- **Full space** (`STAT_COUNT` dims) — used by the ingredient filter.
- **Projected space** (`stat_count` = number of *active* stats) — used by the
  search engine and Pareto cull. Suffix `_proj` everywhere means "projected".

---

## `build_query(...)`

```python
build_query(
    user_json: dict,
    search_for_inversion: bool,
    item_type: str | None = None,
    skill: str | None = None,
    consumable: bool = False,
    fast_cull: bool = False,
    full_meta: bool = False,
) -> Query
```

| param | meaning |
|-------|---------|
| `user_json` | stat-name → config dict (see below); optional `"_context"` key for build context |
| `search_for_inversion` | include negative-effectiveness arrangements in the search. Changes cull soundness (see `fast_cull`). |
| `item_type` / `skill` | profession filters used by the ingredient filter |
| `consumable` | `True` for COOKING/SCRIBING/ALCHEMISM → duration/charges instead of durability |
| `fast_cull` | legacy single-representative ingredient cull. Faster, **unsound under inversion**. Off by default. |
| `full_meta` | enable the extra all-6-meta recovery pass after the normal search |

### Per-stat config dict

```jsonc
"mr":       { "min": 0, "weight": 1000, "ingredient_filter": true },
"strReq":   { "max": 45 },
"hpBonus":  { "min": 2000, "base": 500 }
```

- `min` / `max` — hard constraints (reject a build outside them) **and** filter hints.
- `weight` — objective coefficient. **Sign carries intent**: positive = higher is
  better, negative = lower is better. `weight: 0` means "active but not scored".
- `base` / `min_base` / `max_base` — additive baseline for **non-requirement**
  stats. For **requirement** stats `base` is captured separately and combined with
  the craft value via **MAX** (not sum) — modelling "the player allocates the
  larger of the two". `base` > a user `max` raises `ValueError`. **Forbidden on
  composites** (ambiguous which dep it targets).
- `ingredient_filter` — whether this stat feeds the ingredient filter. Defaults to
  `True` when any of min/max/weight is set; explicit value wins and is *not*
  overwritten by composite propagation.

Unknown stat names are silently skipped. The dict is parsed in two passes: pass 1
handles base stats, pass 2 handles composites (so their deps are activated after).

---

## Composites (derived stats)

If a key is in `DERIVED_DEPENDENCIES`, it's a composite (spell/ehp/ehpr/hpr). Pass 2:
- Resolves the formula tag (string → `_FORMULA_TAGS`; or `("spell", id)` →
  `FORMULA_SPELL_DAMAGE_BASE + id`), validating fixed-formula arity against the dep
  count (spells skip the arity check).
- **Activates every dependency** (sets its active bit; applies `DEFAULT_BASE` if no
  explicit user base). This guarantees the dep projection never yields `-1`.
- Propagates the composite's `ingredient_filter` to its deps (respecting explicit
  per-dep overrides).

Composites are stored as a **struct-of-arrays** for numba (all length `comp_count`):
`comp_formula`, `comp_dep_offset`, `comp_dep_count`, `comp_dep_indices` (flat,
projected), `comp_min/max`, `comp_has_min/has_max`, `comp_weight`. For composite
`c`, its deps are `comp_dep_indices[offset : offset+count]` in the formula's
canonical order.

---

## Key `Query` fields

### Full space (ingredient filter)
`min_stats`, `max_stats` (int32), `weights` (f32), `has_min_mask`, `has_max_mask`,
`filter_mask` (all bool, `STAT_COUNT`), plus `active_indices` (sorted active idxs)
and `proj_stats_idx` (full→proj inverse map, `-1` if inactive).

### Projected space (search)
`min_proj`, `max_proj`, `weights_proj`, `has_min_mask_proj`, `has_max_mask_proj`,
`pos_weight_mask_proj` (w>0), `neg_weight_mask_proj` (w<0),
`base_min_stats_proj` / `base_max_stats_proj`, `stat_count`,
`stat_index_keys_proj` (names).

### Requirements & scoring metadata
- `req_mask_proj` — which projected stats are `*Req`.
- `req_idx` — `(5,)` projected indices of the 5 reqs (or `-1`).
- `round_offset_proj` — `50` for req stats, `0` otherwise. Added before `//100` in
  eff scaling so requirements round like JS `Math.round` (itemIDs) while rolled ids
  floor. Without it, xReq `max` constraints leak by ~6 on a 6-void build.
- `sp_score_*_proj` — skillpoint-cap-aware scoring helpers (str/dex/int/def/agi),
  carrying the ctx base and the linked req index/base used to compute the 150-point cap.

### Cull direction — `lower_better_proj` (the load-bearing one)
`(stat_count,)` bool driving Pareto dominance direction per stat. Decided by
**user intent**, in priority order:

1. `weight > 0` → `False` (higher better)
2. `weight < 0` → `True` (lower better)
3. only `min` set → `False`
4. only `max` set → `True`
5. both / neither → legacy fallback `is_req` (`True` for reqs, else `False`)

This replaced an older "reqs are *always* lower-better" rule. The new rule lets a
positively-weighted requirement (`"defReq": {"weight": +100}`) keep high-req
ingredients at runtime. **Caveat:** the *offline* precalc still bakes the legacy
"reqs lower-better" assumption (see [05](05-precalc-and-meta-sets.md) and session
memory `project_cull_direction_fix`), so positively-weighted-req queries are still
limited by what was precomputed.

### Other
- `suggested_max_cull` — hardcoded **3**. The runtime meta-row cull only runs on
  META_1..3; META_4/5 pass through because culling them costs more than it saves
  (session memory `project_cull_policy`).
- `dura_proj_idx` — projected index of durability (or duration for consumables), or `-1`.
- `build_ctx` — `(18,) f64` query-level constants from `_context`: base skillpoints
  (clamped 0..150), attack-speed multiplier, crit %, per-element weapon damage, and
  the player's base requirement allocation. Passed verbatim to the spell kernels.

---

## Gotchas

- `active_indices` is sorted; `proj_stats_idx[active_indices[j]] == j` is the stable
  inverse used everywhere.
- When `comp_count == 0`, all `comp_*` arrays are explicit empty 1-D arrays (not
  `None`) so numba's typed signatures hold.
- A composite that is active but has no min/max/weight is dropped as inert.
- See session memory `project_composite_known_issues` for a `DEFAULT_BASE` skip bug
  on explicit `min_base`/`max_base` and the `inject_stat` zero-sentinel quirk.
