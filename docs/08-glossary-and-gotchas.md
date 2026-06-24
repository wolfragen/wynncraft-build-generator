# 08 — Glossary & Gotchas

Skim this early; come back when something surprises you.

## Glossary

| Term | Meaning |
|------|---------|
| **Recipe** | The base item being crafted (chestplate, ring, potion…). Carries base durability/duration and `healthOrDamage`, scaled by material **tier**. |
| **Ingredient** | One of up to 6 items placed in the 2×3 grid. Has a stat roll range `[min,max]`, requirements, durability cost, and optional `posMods`. |
| **Slot / void slot** | One of the 6 grid positions (`0..5`). A *void* slot is one left to be filled at runtime by a normal ingredient. |
| **Effectiveness (eff)** | Per-slot scaling % (starts at 100). Stat contribution = `floor(value · eff / 100)` (reqs round instead). Can be **negative** under inversion. |
| **posMods** | An ingredient's per-direction effectiveness modifiers: `above/under` (whole column), `left/right` (single cell), `touching` (4-neighbours), `notTouching` (the rest). |
| **Meta ingredient** | Has `posMods` / `ingredEff` / charges / positive durability. Arrangement matters → enumerated **offline**. |
| **Normal ingredient** | No posMods. Doesn't change anyone's eff → searched **at runtime** to fill void slots. |
| **META_n** | Precomputed dataset: `n` fixed meta ingredients + `6−n` void slots. `n = 1..5`; `META_0` is the all-void synthetic row. |
| **Inversion** | Allowing negative slot effectiveness (`search_for_inversion=True`). Flips the sign of contributions and swaps min/max bounds. |
| **Composite / derived stat** | A stat computed from base stats: `hpr`, `ehp`, `ehpr`, and 13 spell damages. Non-linear; bounded by monotone-corner evaluation in B&B. |
| **Projected space** | The dense array view over only the *active* stats of a query (suffix `_proj`). Contrast with the full `STAT_COUNT` space used by the filter. |
| **Cull** | Pareto-dominance pruning of meta-rows / ingredients to shrink the search. Offline (precalc) and runtime (per-query) variants exist. |
| **Score** | `Σ weight·(0.99·max_roll + 0.01·min_roll)` over stats + composites. |

## Cross-cutting conventions that bite

1. **Three implementations of the craft math must agree.** The effectiveness +
   stat-scaling arithmetic lives in `precalc_fast._grid_kernel` (offline),
   `core/search_engine` (DFS), and `main_decode.compute_*` (oracle). A change in one
   needs the same change in the others. Ground truth: `craft.js`.

2. **Rounding differs by stat kind.** Rolled `ids` use **floor** (`value·eff//100`);
   requirements use **round** (the `+50` `round_offset_proj` before `//100`, matching
   JS `Math.round` on itemIDs). Durability/duration tier-scale with round-half-up
   (`floor(x·mult+0.5)`) while `healthOrDamage` uses floor only.

3. **Negative eff swaps min/max.** Any code scaling a stat by eff must swap the
   bounds when `eff < 0` so `max ≥ min` stays true. The precalc JSON stores the *raw*
   (unswapped) convention; `meta_set_loader`'s cache builder applies the swap. Don't
   double-apply or skip it.

4. **`healthOrDamage` → `hpBonus` for armor, `weapon_dam_neutral` for weapons.** Never
   fold weapon neutral damage into ingredient `nDamRaw` (past ~6× over-valuation bug,
   memory `reference_recipe_matmult`).

5. **Requirements: MAX, not sum.** A player's base requirement allocation combines
   with the craft's via `max(...)`, not addition (one allocation satisfies both).

6. **"Reqs are lower-better" is baked into the *offline* precalc** (`is_req_global`),
   even though the *runtime* cull now respects `lower_better_proj`/user intent. A
   positively-weighted-req query is therefore limited by what was precomputed
   (memory `project_cull_direction_fix`).

7. **Void slots are force-filled.** There is no "empty" sentinel; the DFS always
   fills every void slot, and the useful-mask can prune the least-harmful fill —
   together they can force a *worse-than-empty* fill (caveat #3). `full_meta` only
   sidesteps this for all-6-meta optima.

7b. **Normal-ingredient cull is now a *dual database* (per eff sign).** The scalar
   search no longer feeds one culled list to the kernels. `split_and_cull_by_sign`
   builds a positive-eff DB and a negative-eff DB; `build_dual_db` concatenates them
   (`db.pos_count` marks the boundary) and each void slot iterates only its sign's
   region. This replaced the old unsound `search_inv` range-containment cull (which
   dropped tight high-roll ingredients like `xpb [7,8]`). The membership rule:
   positive-roll ingredients only help a *negative*-weighted stat at *negative* eff,
   etc. See [05](05-precalc-and-meta-sets.md) "The dual database". `pareto_search.py`
   still uses the single-DB API.

8. **Self-recursive `@njit(cache=True)` needs an eager signature** or it segfaults
   (`_DFS_SIG`; memory `project_numba_cache_recursion`). `Date.now`-style nondeterminism
   isn't the issue here — the self-reference symbol is.

9. **Spell `spell_id` order is a contract.** It indexes `SPELLS`, the `("spell", N)`
   formula tags, and cached numba modules. Append-only; never reorder.

10. **Validate search changes by crafter-output diff on an identical query vs.
    single-thread** — not file-level diffs, not against the pre-bugfix archive
    (memory `feedback_cull_comparison`, `project_final_validation`).

11. **`main_decode`'s score excludes spell/ehp/ehpr composites** (no build context).
    Read its printed "notes". It's a faithful total only for queries dominated by
    base stats + `mul_div_100`/`raw_to_pct`.

12. **The search is not guaranteed globally optimal.** Authoritative caveat list:
    [`../README.md`](../README.md) #1–#7. When a competitor build beats ours, suspect
    (in order): an ingredient dropped by the *normal* filter/cull (the dual-DB cull is
    sound now, but the keep-filter still drops, e.g., a stat that has
    `ingredient_filter:False` as its only use — memory `project_borange_bug_lead`), an
    arrangement dropped by the *offline* cull (esp. pure-effectiveness meta ingredients
    like *Borange Fluff* with no stats), the all-6-meta gap (#1), or the
    force-fill/useful-mask interaction (#3) — before suspecting the scorer.

## Environment / repo

- Run from `python/`. Use the conda **`AI`** env
  (`/c/Users/quent/anaconda3/envs/AI/python.exe`), not system Python (memory
  `user_python_env`).
- Only `python/` is a git repo; the parent dir is not (memory `reference_git_layout`).
- Ignore the stale `python - Copie/` backup.
- `wynnbuilder.github.io-master/` is the reference JS implementation (ground truth).
