# 07 — Crafter-URL Encoding, Decode Tool & Entrypoints

Source: `utils/hash_generator.py`, `main_decode.py`, `main.py`, `main_build_temp.py`.

---

## `utils/hash_generator.py` — crafter-URL encoding

Encodes a craft (6 ingredient ids + recipe id + material tier + weapon attack
speed) into a wynnbuilder-compatible base64 hash. It is the **inverse** of the
decoder in `main_decode.py`; the two must stay in lockstep.

`generate_crafter_url(recipe_id, recipe_type, tier, ingredient_ids, atk_speed="NORMAL", base_url=…) -> str`.
Ids are **JSON ids**, not DB row indices. `recipe_type` decides (case-insensitively)
whether attack-speed bits are appended (weapons only). `tier ∈ {1,2,3}` encoded as
`tier-1`.

### Bit layout (`_CRAFTED_ENCODING_VERSION = 2`)
| field | bits | notes |
|-------|------|-------|
| legacy flag | 1 | always `0` |
| version | 7 | `2` |
| ingredient ids | 6 × 12 | `_NUM_INGS=6`, `_ING_ID_BITLEN=12` |
| recipe id | 12 | `_RECIPE_ID_BITLEN=12` |
| material tier ×2 | 2 × 3 | `_NUM_MATS=2`, `_MAT_TIER_BITLEN=3`, value `tier-1` |
| attack speed | 4 | weapons only; `ATTACK_SPEED_MAP = {SLOW:0, NORMAL:1, FAST:2}` |
| padding | to next ×6 | zero bits |

- `CHARSET = "0123456789ABCD…xyz+-"` — non-standard base64 ordering matching
  wynnbuilder's `Base64`. Bits pack little-endian within each 6-bit group.

---

## `main_decode.py` — decode + verify (the oracle)

Decodes any crafter URL and **recomputes the full crafted stats from first
principles**, mirroring `craft.js`. Given a query (same shape as `main.py`), it
reports the weighted score and a VALID/INVALID verdict with the list of violations.
This is the ground-truth check for "is our build actually correct / is theirs really
better?".

What it reproduces faithfully:
- **Effectiveness** from posMods (`compute_effectiveness`) — same column/cell rules
  as [01](01-architecture.md).
- **Final stats** (`compute_crafted_stats`): rolled `ids` floor `value*eff`; req
  `itemIDs` use `Math.round`; durability/duration are tier-scaled recipe base + sum,
  no eff; **armor recipe HP folded into `hpBonus`** (matches `data/recipe.py`); weapon
  neutral damage reported separately as `nDamBase`. Negative-eff rolls swap min/max.
- **Score** (`score_query`): `Σ w·(0.99·max + 0.01·min)` over base stats + the two
  supported composites (`mul_div_100`, `raw_to_pct`). Composites needing build
  context (spells, ehp/ehpr) are **listed under "notes" and excluded** from the
  decoder's score — it can't reproduce them without `_context`.

### How to use it
Set `URL` and `USER_QUERY` at the bottom of the file and run it, or import
`decode_and_report(url, user_query)`. To compare two URLs under one query, a tiny
driver (see the session's `_compare_decode.py`) calls `decode_and_report` twice.

> ⚠️ The decoder's score covers **base stats + 2 composite formulas only**. If the
> real query is dominated by spell/ehp composites, the decoder's number is a partial
> proxy — read the "notes" it prints. For pure damage-%/raw/skillpoint/mr queries
> (like `main.py`'s sample) it's a faithful total.

---

## Entrypoints

### `main.py` — the primary scalar search
Top of file computes per-element damage **weights** from a weapon-damage model
(`nScale/eScale/…`, the `dps`/`pctW` knobs, `raw()` helper) and builds `user_query`.
Then: load ingredients → `build_query` → `find_recipe`/`build_recipe` →
`filter_raw_ingredients` → `IngredientDB` → `search_pipelined` → print build +
crafter URL. Edit the `user_query`/`skill`/`item_type` block to change the search.
`skill` maps to `item_type` via the `dico` dict.

### `main_pareto.py` — the Pareto-frontier mode
See [06](06-pareto-mode.md).

### `main_build_temp.py` — fill empty slots of an existing build
Decodes a full wynnbuilder **build** URL (9 equipment slots), aggregates the stats
of the already-equipped pieces, then for each empty slot runs `search_pipelined`
with those aggregate stats injected as `base_min/base_max` so the crafted piece
complements the existing gear. (Large file; bit-cursor decoder up top.)
