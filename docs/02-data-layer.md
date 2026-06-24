# 02 — Data Layer

Source: `data/stats.py`, `data/ingredient_loader.py`, `data/ingredient_db.py`,
`data/recipe.py`, `data/recipe_loader.py`, `data/spells.py`,
`data/skillpoint_lookup.py`.

The data layer turns the two compressed JSON files into dense numpy arrays indexed
by a single global stat registry, so every hot path is array math (no dict lookups).

---

## `stats.py` — the global stat registry

Defines every supported stat and flattens them into one dense index space.

- `IDS_STATS` — the "rolled" identification stats (damage %, raw, defs, skillpoints,
  hpr, mr, etc.), sorted.
- `REQ_STATS` — the 5 requirements (`strReq, dexReq, intReq, defReq, agiReq`).
- `SPECIAL_STATS` — `durability`, `duration`, `charges`.
- `ALL_STATS = IDS_STATS + REQ_STATS + SPECIAL_STATS`; `STAT_INDEX` maps name→index;
  `STAT_COUNT = len(ALL_STATS)`. **Adding a stat to `IDS_STATS` automatically
  reindexes everything** — there is no second table to keep in sync.
- `IDX_DURABILITY / IDX_DURATION / IDX_CHARGES` — convenience indices.
- `CONSU_SKILLS = ["ALCHEMISM", "SCRIBING", "COOKING"]` — these craft consumables
  (use `duration`/`charges` instead of `durability`).

### Derived / composite stats

`DERIVED_STATS` lists the composites: `hpr`, `ehp`, `ehpr`, and 13 spell damages
(`mage_meteor`, `warrior_bash`, …). For each:

- `DERIVED_DEPENDENCIES[name]` — the ordered tuple of base stats it reads. Spells
  use one of two **canonical** dep layouts, `_SPELL_DEPS` (len 36) or `_MELEE_DEPS`
  (len 32), mirrored from `data/spells.py`. **These mirrors MUST match `spells.py`**;
  the query parser asserts arity to catch drift.
- `DERIVED_FORMULA[name]` — a formula tag: `"raw_to_pct"`, `"ehp"`, `"ehpr"`,
  `"mul_div_100"`, or `("spell", spell_id)`.
- Numba-friendly int tags: `FORMULA_MUL_DIV_100=0`, `FORMULA_RAW_TO_PCT=1`,
  `FORMULA_EHP=2`, `FORMULA_EHPR=3`, and spells at `FORMULA_SPELL_DAMAGE_BASE(100) + spell_id`.

`DEFAULT_BASE` (currently `{}`) is the hook for "a fresh item already has stat X";
applied when a stat becomes active without an explicit user base.

---

## `ingredient_loader.py` — raw ingredients → dense vectors

`load_ingredients(path) -> list[RawIngredient]`. Each `RawIngredient` (NamedTuple):

| field | shape / type | meaning |
|-------|--------------|---------|
| `ing_id` | int | JSON id |
| `name` | str | |
| `stats_min` / `stats_max` | `(STAT_COUNT,)` int16 | per-stat roll range, dense |
| `skills` | `(8,)` bool | which professions can use it (`SKILL_ORDER`) |
| `pos_mods` | `(6,)` int16 | posMods in `POSMOD_ORDER` |
| `tier`, `lvl`, `ing_type` | int | |

`_build_stat_vectors` walks the three JSON namespaces and maps each into the dense
vector. **Namespace → stat mapping gotchas:**
- `ids.*` → the like-named stat.
- `itemIDs.dura` → **`durability`**; `itemIDs.{x}Req` → the requirement stats.
- `consumableIDs.dura` → **`duration`**; `consumableIDs.charges` → `charges`.

Missing stats default to `0` (both bounds). Order of returned ingredients = JSON
order (no sorting here).

`SKILL_ORDER` / `SKILL_INDEX` (8 skills) and `POSMOD_ORDER` (6 keys) are the fixed
orders everything else relies on.

---

## `recipe_loader.py` — recipes → `RawRecipe`

- `load_recipes(path) -> list[RawRecipe]` reads `recipes_compress.json["recipes"]`.
- `RawRecipe` = `(item_type, skill_index, lvl_min, lvl_max, data)` where `data` is
  the full raw dict (durability, healthOrDamage, basicDuration, materials…).
- `find_recipe(recipes, item_type, skill, lvl_min, lvl_max)` returns the first
  exact match; raises `ValueError` on unknown skill / no match.

---

## `recipe.py` — `build_recipe(raw, query, tier) -> Recipe`

Parses one recipe and injects its tier-scaled base stats into the query's
**projected** stat space.

`TIER_MULT = (0.0, 1.0, 1.25, 1.4)` — index by tier (1..3). `Recipe` fields:

| field | meaning |
|-------|---------|
| `base_min_stats_proj` / `base_max_stats_proj` | `(query.stat_count,)` base stats projected onto active dims, with recipe contributions injected |
| `scaled_dura_min` / `scaled_dura_max` | durability (or duration) after the tier multiplier |
| `weapon_dam_neutral` | `(min,max)` neutral weapon damage — **weapons only**, else `(0,0)` |

### Two scaling conventions that differ — don't "fix" one to match the other

- **Durability / duration**: `floor(base * mult + 0.5)` → round-half-up, matching
  JS `Math.round`.
- **healthOrDamage**: `floor(base * mult)` → floor only. Different on purpose.

### `healthOrDamage` injection — armor vs weapon

- **Armor** (ARMOURING, TAILORING): `healthOrDamage` is HP → injected into the
  **`hpBonus`** stat. (The decoder mirrors this: `main_decode` folds recipe HP into
  `hpBonus`.)
- **Weapons**: `healthOrDamage` is neutral weapon damage → stored in
  `weapon_dam_neutral`, **not** folded into ingredient `nDamRaw`. The search reads
  it as `ctx.weapon_dam` for spell scaling. (A past bug folded it into `nDamRaw`,
  multiplying ingredient raw-damage rolls by weapon multipliers and ~6× over-valuing
  them — see session memory `reference_recipe_matmult`.)

`inject_stat` intersects rather than sums when a base already has a value (recipe
range narrows the active range). See session memory `project_composite_known_issues`
for an `inject_stat` zero-sentinel quirk.

---

## `ingredient_db.py` — the compact search DB

`IngredientDB(filtered_ingredients, query)` projects each surviving normal
ingredient onto the **active** stat dims and **sorts by estimated score
contribution descending**, so the DFS visits high-impact ingredients first (tighter
`best_score` earlier → harder branch-and-bound pruning).

Key attributes (all row-aligned, sorted):
- `stat_min_matrix` / `stat_max_matrix` — `(count, stat_count)` int16, projected rolls.
- `json_ids` — `(count,)` int32, original JSON id per row.
- `contrib_pos_mask` (`stat_max>0`) / `contrib_neg_mask` (`stat_min<0`) — `(count, stat_count)` bool.

The sort heuristic adds each composite's weight onto every dependency stat, then
`Σ positive_weight·stat_max + Σ negative_weight·stat_min`. Crude (composites are
non-linear) but cheap and roughly right.

---

## `spells.py` — spell metadata

Encodes per-spell element multipliers and damage mode, plus numpy arrays for the
kernel. Each of the 13 `SPELLS` entries has `mults` (per-element % conversion),
`use_spell` (spell-damage vs melee-damage scaling → which canonical dep layout),
and `ignore_speed`.

- `SPELL_SPELL_DEPS` (36) and `SPELL_MELEE_DEPS` (32) are the canonical dep orders
  (mirrored in `stats.py`). `get_spell_deps(spell_id)` returns the right one.
- Derived arrays: `SPELL_MULTS (13,6) int32`, `SPELL_TOTAL_CONVERT (13,) f64`,
  `SPELL_USE_SPELL`, `SPELL_IGNORE_SPEED`, `SPELL_INDEX`.
- `ATK_SPEED_MULT` — base damage multiplier per attack speed (from
  wynnbuilder `build_utils.js`).

**`spell_id = SPELLS.index(...)` is a stable contract.** Appending spells is safe;
reordering invalidates cached numba modules and the `("spell", N)` formula tags.

See session memory `project_spell_formula_redesign` for the single generic
evaluator + canonical 31/27 dep-layout design.

---

## `skillpoint_lookup.py` — the SP→% curve

Skillpoints (str/dex/int/def/agi, 0..`SKP_MAX=150`) convert non-linearly to
sub-stat percentages (wynnbuilder's `skillPointsToPercentage`). Because the curve
is non-linear, **it is only valid on a build's total SP count, never per-ingredient**.

Precomputed lookup tables (built once at import):
- `SKP_PCT_BASE (151,) f64` — base curve value per SP count.
- `SKP_HEADLINE_PCT (5,151) f64` — headline effect (str→%dmg, dex→%crit,
  int→%cost-reduction capped at 50%, def→%def-reduction, agi→%dodge).
- `SKP_ELEMENT_PCT (5,151) f64` — the elemental-damage % each SP grants.

`lookup_headline(skp_idx, count)` / `lookup_element(skp_idx, count)` clamp the
count to `[0,150]` and index the tables. The int multiplier `≈0.619` is chosen so
the headline tops out at exactly 50% mana-cost reduction.

---

## On-disk JSON formats

### `data/ingreds_compress.json` — array of ingredient objects

```jsonc
{
  "id": 858, "name": "Forsaken Catalyst", "tier": 2, "lvl": 119,
  "skills": ["ARMOURING", "TAILORING"],
  "ids":          { "spd": {"minimum":4,"maximum":6}, "aDamRaw": {"minimum":25,"maximum":30} },
  "itemIDs":      { "dura": -124, "strReq": 0, "dexReq": -20, "agiReq": 20 },   // dura=durability
  "consumableIDs":{ "dura": 0, "charges": 0 },                                  // dura=duration
  "posMods":      { "left": -53, "right": -53, "above": 0, "under": -53, "touching": 53, "notTouching": 0 }
}
```
- `ids` values are scalars or `{minimum,maximum}`. `posMods` non-zero ⇒ **meta** ingredient.

### `data/recipes_compress.json` — `{ "recipes": [ … ] }`

```jsonc
{
  "id": 43, "name": "Boots-1-3", "type": "BOOTS", "skill": "TAILORING",
  "lvl":            {"minimum":1,"maximum":3},
  "durability":     {"minimum":175,"maximum":182},
  "healthOrDamage": {"minimum":9,"maximum":11},   // HP for armor, neutral dmg for weapons
  "materials":      [ {"item":"Refined Copper Ingot","amount":1}, … ],
  "basicDuration":  {…}                            // optional, consumables → charges/duration
}
```

### `data/precalc/.../META_n.json` — see [05](05-precalc-and-meta-sets.md).
