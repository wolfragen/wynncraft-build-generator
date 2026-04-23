"""
precalc_fast.py

Drop-in faster replacement for precalc.run_precalculation. Output is
designed to be byte-equivalent to the original Python pipeline:

  - Same JSON layout (`{"ings": [...], "eff": [...], "stats": {...}}`).
  - Same per-stat insertion order in the `stats` dict
    (slot 0..5; per ingredient: ids in JSON order, then reqs in fixed
    order, then static dura/charges).
  - Same per-slot dura prune.
  - Same per-multiset local_cull semantics:
        * eff comparison: drop zeros; sort positives desc, negatives asc;
          incomparable when counts differ; otherwise zip-wise.
        * stat comparison: req → lower min/max better, else → higher better.
        * order: process recipes in generation order; survivors-list Pareto.

The hot paths (grid-stat compute, dura prune, Pareto compare) are numba JIT.
The orchestration is plain Python so multiprocessing wraps it cleanly.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import time
from typing import List, Tuple

import numpy as np
from numba import njit

# Default to compact orjson output (smaller files, faster write).
# Set PRECALC_FAST_LEGACY_JSON=1 to fall back to stdlib json.dumps with
# the legacy precalc.py formatting (spaces after `,` and `:`).
try:
    import orjson
    _HAVE_ORJSON = True
except ImportError:
    _HAVE_ORJSON = False
_USE_ORJSON = _HAVE_ORJSON and not bool(int(os.environ.get('PRECALC_FAST_LEGACY_JSON', '0')))

from data.recipe_loader import load_recipes
from data.stats import CONSU_SKILLS

# ---------------------------------------------------------------------------
# Constants describing the 2x3 grid posMod targets, exactly as in
# precalc.calculate_recipe_stats.
#
# posmod order per ingredient: ['left','right','above','under','touching','notTouching'] (index 0..5)
# For each source slot i and posmod p, TARGETS[i, p, :COUNTS[i, p]] are the
# target slots that get += pos_mods[f, p].
# ---------------------------------------------------------------------------

def _build_target_tables() -> Tuple[np.ndarray, np.ndarray]:
    # Adjacency map taken verbatim from precalc.py.
    adjacents = {
        0: [1, 2], 1: [0, 3], 2: [0, 3, 4],
        3: [1, 2, 5], 4: [2, 5], 5: [3, 4],
    }
    targets_list = [[[] for _ in range(6)] for _ in range(6)]
    for i in range(6):
        # left/right (single-cell, mirroring the original Python)
        if i % 2 == 0:
            targets_list[i][1].append(i + 1)  # right
        else:
            targets_list[i][0].append(i - 1)  # left
        # above/under: same column
        column = [0, 2, 4] if i % 2 == 0 else [1, 3, 5]
        for target in column:
            if target > i:
                targets_list[i][3].append(target)  # under
            elif target < i:
                targets_list[i][2].append(target)  # above
        # touching
        for target in adjacents[i]:
            targets_list[i][4].append(target)
        # notTouching
        for target in range(6):
            if target != i and target not in adjacents[i]:
                targets_list[i][5].append(target)

    max_targets = max(len(targets_list[i][p]) for i in range(6) for p in range(6))
    TARGETS = np.full((6, 6, max_targets), -1, dtype=np.int8)
    COUNTS = np.zeros((6, 6), dtype=np.int8)
    for i in range(6):
        for p in range(6):
            for k, t in enumerate(targets_list[i][p]):
                TARGETS[i, p, k] = t
            COUNTS[i, p] = len(targets_list[i][p])
    return TARGETS, COUNTS


_TARGETS, _COUNTS = _build_target_tables()


# ---------------------------------------------------------------------------
# Ingredient-side preprocessing
# ---------------------------------------------------------------------------

POSMOD_ORDER = ('left', 'right', 'above', 'under', 'touching', 'notTouching')
REQ_NAMES = ('strReq', 'dexReq', 'intReq', 'defReq', 'agiReq')


class FilteredArrays:
    """Dense per-ingredient arrays consumed by the JIT kernel.

    A "stat slot" inside an ingredient corresponds to one (name, scalable?) pair.
    The stats are listed in the same order as the original Python builder
    inserts them into ``ing["stats"]`` / ``ing["static_stats"]``:

        ids (in JSON order, ingredEff included)
        + reqs (strReq, dexReq, intReq, defReq, agiReq, only non-zero)
        + static dura (durability or duration)
        + static charges
    """

    def __init__(self, F: int, max_stats: int, num_global_stats: int):
        self.F = F
        self.max_stats = max_stats
        self.num_global_stats = num_global_stats

        self.ing_ids = np.zeros(F, dtype=np.int32)
        self.ing_pos_mods = np.zeros((F, 6), dtype=np.int32)
        self.ing_ingred_eff = np.zeros(F, dtype=np.int32)

        # Per-ingredient ordered stat list:
        self.ing_stat_idx = np.full((F, max_stats), -1, dtype=np.int32)  # global stat index
        self.ing_stat_min = np.zeros((F, max_stats), dtype=np.int32)
        self.ing_stat_max = np.zeros((F, max_stats), dtype=np.int32)
        self.ing_stat_scalable = np.zeros((F, max_stats), dtype=np.bool_)
        self.ing_stat_count = np.zeros(F, dtype=np.int32)

        # Mapping global stat index -> (name, is_req)
        self.stat_names: List[str] = [''] * num_global_stats
        self.is_req_global = np.zeros(num_global_stats, dtype=np.bool_)


def _filter_and_pack(
    all_ingreds: list,
    profession: str,
    include_dura: bool,
    dura_str: str,
) -> FilteredArrays:
    """Replicates precalc.get_filtered_ingredients but emits dense arrays.

    Returns a FilteredArrays plus implicit ordering identical to the original
    builder, which guarantees byte-identical JSON output for `stats`.
    """

    # First pass: discover per-ingredient stat lists in their canonical order
    # so we can build a global stat name -> index mapping.
    filtered_raw = []
    for ing in all_ingreds:
        if profession not in ing.get('skills', []):
            continue

        pos = ing.get('posMods', {})
        has_pos_mods = any(v != 0 for v in pos.values())
        ids = ing.get('ids', {})
        has_ingred_eff = 'ingredEff' in ids
        item_ids = ing.get('itemIDs', {})
        cons_ids = ing.get('consumableIDs', {})
        charges = cons_ids.get('charges', 0)

        is_meta = has_pos_mods or has_ingred_eff or charges != 0

        durability = item_ids.get('dura', 0)
        duration = cons_ids.get('dura', 0)
        dura = duration if dura_str == 'duration' else durability
        is_dura = dura > 0

        if not (is_meta or (include_dura and is_dura)):
            continue

        filtered_raw.append((ing, pos, ids, item_ids, cons_ids, charges, dura))

    F = len(filtered_raw)

    # Walk every ingredient once to figure out stat insertion order.
    # `stat_to_global` is built lazily as new names appear, in the same order
    # the original Python builder would have first inserted them; it doesn't
    # matter for JSON output (each recipe re-derives its own order at write
    # time), but having a stable global index lets us key the dense arrays.
    stat_to_global: dict = {}
    is_req_global_list: list = []

    def _intern(name: str, is_req: bool) -> int:
        idx = stat_to_global.get(name)
        if idx is None:
            idx = len(stat_to_global)
            stat_to_global[name] = idx
            is_req_global_list.append(is_req)
        return idx

    ordered_per_ing: List[List[Tuple[int, int, int, bool]]] = []
    for ing, pos, ids, item_ids, cons_ids, charges, dura in filtered_raw:
        seq: List[Tuple[int, int, int, bool]] = []

        # ids in JSON order (excluding ingredEff value but ingredEff itself
        # is consumed only to bump the slot's eff, not as a stat — matching
        # `if name == "ingredEff": continue` in calculate_recipe_stats).
        for name, val in ids.items():
            if name == 'ingredEff':
                continue
            if isinstance(val, dict):
                vmin = val.get('minimum', 0)
                vmax = val.get('maximum', 0)
            else:
                vmin = val
                vmax = val
            seq.append((_intern(name, name in REQ_NAMES), vmin, vmax, True))

        # reqs (fixed order), only when present (non-zero)
        for r in REQ_NAMES:
            v = item_ids.get(r, 0)
            if v != 0:
                # Original puts these into "stats" so they ARE multiplied by eff.
                seq.append((_intern(r, True), v, v, True))

        # static dura
        if dura != 0:
            seq.append((_intern(dura_str, False), dura, dura, False))

        # static charges
        if charges != 0:
            seq.append((_intern('charges', False), charges, charges, False))

        ordered_per_ing.append(seq)

    num_global_stats = len(stat_to_global)
    max_stats = max((len(s) for s in ordered_per_ing), default=0)

    arrays = FilteredArrays(F, max_stats, num_global_stats)
    for f, ((ing, pos, ids, item_ids, cons_ids, charges, dura), seq) in enumerate(
        zip(filtered_raw, ordered_per_ing)
    ):
        arrays.ing_ids[f] = int(ing['id'])
        for p, key in enumerate(POSMOD_ORDER):
            arrays.ing_pos_mods[f, p] = pos.get(key, 0)
        # ingredEff: only the .min field is used by the original to boost the slot's eff.
        ie = ids.get('ingredEff', None)
        if ie is None:
            arrays.ing_ingred_eff[f] = 0
        elif isinstance(ie, dict):
            arrays.ing_ingred_eff[f] = ie.get('minimum', 0)
        else:
            arrays.ing_ingred_eff[f] = int(ie)
        for kk, (gidx, vmin, vmax, scalable) in enumerate(seq):
            arrays.ing_stat_idx[f, kk] = gidx
            arrays.ing_stat_min[f, kk] = vmin
            arrays.ing_stat_max[f, kk] = vmax
            arrays.ing_stat_scalable[f, kk] = scalable
        arrays.ing_stat_count[f] = len(seq)

    for name, idx in stat_to_global.items():
        arrays.stat_names[idx] = name
    for idx, is_req in enumerate(is_req_global_list):
        arrays.is_req_global[idx] = is_req

    return arrays


# ---------------------------------------------------------------------------
# JIT kernel: per-grid eff + stats + dura prune
# ---------------------------------------------------------------------------

@njit(cache=True)
def _grid_kernel(
    grid,                  # (6,) int32, -1 for void
    ing_pos_mods,          # (F, 6) int32
    ing_ingred_eff,        # (F,) int32
    ing_stat_idx,          # (F, max_stats) int32
    ing_stat_min,          # (F, max_stats) int32
    ing_stat_max,          # (F, max_stats) int32
    ing_stat_scalable,     # (F, max_stats) bool
    ing_stat_count,        # (F,) int32
    TARGETS,               # (6, 6, max_targets) int8
    COUNTS,                # (6, 6) int8
    num_global_stats,      # int
    out_eff,               # (6,) int32
    out_keys_in_order,     # (num_global_stats,) int32
    out_total_min,         # (num_global_stats,) int32
    out_total_max,         # (num_global_stats,) int32
    seen_scratch,          # (num_global_stats,) bool — must be reset by caller or here
):
    # Reset working state.
    for s in range(6):
        out_eff[s] = 100
    for k in range(num_global_stats):
        seen_scratch[k] = False
        out_total_min[k] = 0
        out_total_max[k] = 0

    # ---- compute effs ----
    for i in range(6):
        f = grid[i]
        if f == -1:
            continue
        out_eff[i] += ing_ingred_eff[f]
        for p in range(6):
            v = ing_pos_mods[f, p]
            if v == 0:
                continue
            cnt = COUNTS[i, p]
            for k in range(cnt):
                out_eff[TARGETS[i, p, k]] += v

    # ---- accumulate stats in insertion order ----
    key_count = 0
    for i in range(6):
        f = grid[i]
        if f == -1:
            continue
        e = out_eff[i]
        n = ing_stat_count[f]
        for kk in range(n):
            k = ing_stat_idx[f, kk]
            if ing_stat_scalable[f, kk]:
                b_min = (ing_stat_min[f, kk] * e) // 100
                b_max = (ing_stat_max[f, kk] * e) // 100
                if b_min == 0 and b_max == 0:
                    continue  # mirror per-slot zero skip
            else:
                b_min = ing_stat_min[f, kk]
                b_max = ing_stat_max[f, kk]

            if not seen_scratch[k]:
                seen_scratch[k] = True
                out_keys_in_order[key_count] = k
                key_count += 1
            out_total_min[k] += b_min
            out_total_max[k] += b_max

    return key_count


# ---------------------------------------------------------------------------
# JIT kernel: pack a recipe into per-multiset arrays for Pareto
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pack_for_pareto(
    eff,                  # (6,) int32 — full eff vector (placed slots included)
    grid,                 # (6,) int32
    keys_in_order,        # (num_global_stats,)
    total_min,            # (num_global_stats,)
    total_max,            # (num_global_stats,)
    key_count,            # int
    is_req_global,        # (num_global_stats,) bool
    out_pos,              # (6,) int32 — positive void effs sorted desc, zero-padded
    out_neg,              # (6,) int32 — negative void effs sorted asc, zero-padded
    out_pos_count,        # scalar -> returned
    out_neg_count,        # scalar -> returned
    out_stat_min,         # (num_global_stats,) — gets totals indexed by global stat
    out_stat_max,
    out_stat_present,     # (num_global_stats,) bool — True iff this recipe has the stat
):
    # Effs of empty (void) slots only.
    pos_n = 0
    neg_n = 0
    tmp_pos = np.zeros(6, dtype=np.int32)
    tmp_neg = np.zeros(6, dtype=np.int32)
    for i in range(6):
        if grid[i] != -1:
            continue
        e = eff[i]
        if e > 0:
            tmp_pos[pos_n] = e
            pos_n += 1
        elif e < 0:
            tmp_neg[neg_n] = e
            neg_n += 1

    # Sort: positives descending, negatives ascending.
    # Insertion sort — n ≤ 6 so this is faster than np.sort overhead.
    for i in range(1, pos_n):
        v = tmp_pos[i]
        j = i - 1
        while j >= 0 and tmp_pos[j] < v:
            tmp_pos[j + 1] = tmp_pos[j]
            j -= 1
        tmp_pos[j + 1] = v
    for i in range(1, neg_n):
        v = tmp_neg[i]
        j = i - 1
        while j >= 0 and tmp_neg[j] > v:
            tmp_neg[j + 1] = tmp_neg[j]
            j -= 1
        tmp_neg[j + 1] = v

    for k in range(6):
        out_pos[k] = tmp_pos[k] if k < pos_n else 0
        out_neg[k] = tmp_neg[k] if k < neg_n else 0

    # Stats: scatter totals into dense arrays indexed by global stat.
    G = total_min.shape[0]
    for k in range(G):
        out_stat_present[k] = False
        out_stat_min[k] = 0
        out_stat_max[k] = 0
    for kk in range(key_count):
        k = keys_in_order[kk]
        out_stat_present[k] = True
        out_stat_min[k] = total_min[k]
        out_stat_max[k] = total_max[k]

    return pos_n, neg_n


# ---------------------------------------------------------------------------
# JIT kernel: compare two recipes (precalc.compare_recipes semantics)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _compare_recipes_jit(
    pos_a, pos_b, pos_n_a, pos_n_b,
    neg_a, neg_b, neg_n_a, neg_n_b,
    stat_min_a, stat_min_b, stat_max_a, stat_max_b,
    stat_present_a, stat_present_b,
    is_req_global,
):
    """Returns 1 if A dominates B, -1 if B dominates A, 2 if identical, 0 if incomparable."""

    if pos_n_a != pos_n_b or neg_n_a != neg_n_b:
        return 0

    a_better = False
    b_better = False

    for k in range(pos_n_a):
        pa = pos_a[k]
        pb = pos_b[k]
        if pa > pb:
            a_better = True
        elif pb > pa:
            b_better = True
    for k in range(neg_n_a):
        na = neg_a[k]
        nb = neg_b[k]
        if na < nb:
            a_better = True
        elif nb < na:
            b_better = True

    if a_better and b_better:
        return 0

    G = stat_min_a.shape[0]
    for k in range(G):
        # Match `set(a.keys()).union(b.keys())`: skip stats absent on both sides.
        if not stat_present_a[k] and not stat_present_b[k]:
            continue
        a_min = stat_min_a[k]
        a_max = stat_max_a[k]
        b_min = stat_min_b[k]
        b_max = stat_max_b[k]

        if is_req_global[k]:
            # Requirements: lower is better.
            if a_min < b_min:
                a_better = True
            elif b_min < a_min:
                b_better = True
            if a_max < b_max:
                a_better = True
            elif b_max < a_max:
                b_better = True
        else:
            # Standard stats: higher is better.
            if a_min > b_min:
                a_better = True
            elif b_min > a_min:
                b_better = True
            if a_max > b_max:
                a_better = True
            elif b_max > a_max:
                b_better = True

        if a_better and b_better:
            return 0

    if a_better:
        return 1
    if b_better:
        return -1
    return 2


# ---------------------------------------------------------------------------
# JIT kernel: per-multiset local cull (mirrors precalc.local_cull semantics)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _local_cull_jit(
    n_recipes,
    pos_arr, neg_arr, pos_n_arr, neg_n_arr,
    stat_min_arr, stat_max_arr, stat_present_arr,
    is_req_global,
    survivor_indices,  # output, capacity n_recipes
):
    """Returns count of survivors written into survivor_indices."""
    n_surv = 0
    survivor_alive = np.ones(n_recipes, dtype=np.bool_)  # tracks current survivors slot-wise
    survivor_slot = np.zeros(n_recipes, dtype=np.int32)  # mapping survivor index -> recipe index

    for r in range(n_recipes):
        is_dominated = False
        # Iterate current survivors, marking ones that r dominates for removal.
        s = 0
        while s < n_surv:
            sidx = survivor_slot[s]
            cmp = _compare_recipes_jit(
                pos_arr[r], pos_arr[sidx], pos_n_arr[r], pos_n_arr[sidx],
                neg_arr[r], neg_arr[sidx], neg_n_arr[r], neg_n_arr[sidx],
                stat_min_arr[r], stat_min_arr[sidx],
                stat_max_arr[r], stat_max_arr[sidx],
                stat_present_arr[r], stat_present_arr[sidx],
                is_req_global,
            )
            if cmp == -1 or cmp == 2:
                # Existing survivor strictly dominates r, OR they're identical.
                is_dominated = True
                break
            elif cmp == 1:
                # r strictly dominates this survivor: remove it (swap-pop preserves
                # remaining order if we shift; original code used reversed-pop, which
                # also preserves the survivor list order. Use a shift-down to keep order.
                for t in range(s, n_surv - 1):
                    survivor_slot[t] = survivor_slot[t + 1]
                n_surv -= 1
                continue
            s += 1

        if not is_dominated:
            survivor_slot[n_surv] = r
            n_surv += 1

    for i in range(n_surv):
        survivor_indices[i] = survivor_slot[i]
    return n_surv


# ---------------------------------------------------------------------------
# Python orchestration: per-multiset processing
# ---------------------------------------------------------------------------

def _unique_perms(items):
    """Yield unique permutations by ingredient-array index (matches precalc.get_unique_permutations)."""
    seen = set()
    for perm in itertools.permutations(items):
        if perm not in seen:
            seen.add(perm)
            yield perm


def _process_multiset(
    multiset_indices: tuple,
    n: int,
    arrays: FilteredArrays,
    should_check_dura: bool,
    dura_global_idx: int,
    min_dura: int,
    # reusable scratch buffers
    grid_buf, eff_buf, keys_buf, total_min_buf, total_max_buf, seen_buf,
    pos_arr, neg_arr, pos_n_arr, neg_n_arr,
    stat_min_arr, stat_max_arr, stat_present_arr,
    survivor_indices_buf,
):
    """Returns (recipes_for_this_multiset, count_generated, count_culled).

    Each recipe is a fully-built dict ready to JSON-dump.
    """

    n_recipes = 0
    n_generated = 0
    n_dura_culled = 0
    G = arrays.num_global_stats

    # Slot-index combos: iterate over choosing n positions out of 6.
    # Keep a stable list to avoid recomputing inside.
    slot_combos = list(itertools.combinations(range(6), n))

    keys_in_order_per_recipe: List[Tuple[np.ndarray, int, np.ndarray]] = []
    grid_per_recipe: List[np.ndarray] = []
    eff_per_recipe: List[np.ndarray] = []

    for perm in _unique_perms(multiset_indices):
        for active_slots in slot_combos:
            grid_buf[:] = -1
            for k, gidx in enumerate(active_slots):
                grid_buf[gidx] = perm[k]

            key_count = _grid_kernel(
                grid_buf,
                arrays.ing_pos_mods, arrays.ing_ingred_eff,
                arrays.ing_stat_idx, arrays.ing_stat_min, arrays.ing_stat_max,
                arrays.ing_stat_scalable, arrays.ing_stat_count,
                _TARGETS, _COUNTS,
                G,
                eff_buf, keys_buf, total_min_buf, total_max_buf, seen_buf,
            )
            n_generated += 1

            if should_check_dura and dura_global_idx >= 0 and seen_buf[dura_global_idx]:
                if total_max_buf[dura_global_idx] < min_dura:
                    n_dura_culled += 1
                    continue

            # Snapshot per-recipe data; pack for Pareto.
            grid_snap = grid_buf.copy()
            eff_snap = eff_buf.copy()
            keys_snap = keys_buf[:key_count].copy()
            tmin_snap = total_min_buf.copy()
            tmax_snap = total_max_buf.copy()

            # Pack into Pareto row.
            row = n_recipes
            pos_n, neg_n = _pack_for_pareto(
                eff_snap, grid_snap,
                keys_snap, tmin_snap, tmax_snap, key_count,
                arrays.is_req_global,
                pos_arr[row], neg_arr[row],
                0, 0,
                stat_min_arr[row], stat_max_arr[row], stat_present_arr[row],
            )
            pos_n_arr[row] = pos_n
            neg_n_arr[row] = neg_n

            grid_per_recipe.append(grid_snap)
            eff_per_recipe.append(eff_snap)
            keys_in_order_per_recipe.append((keys_snap, key_count, tmin_snap))

            n_recipes += 1

    if n_recipes == 0:
        return [], n_generated, n_dura_culled

    # Local Pareto cull.
    n_surv = _local_cull_jit(
        n_recipes,
        pos_arr[:n_recipes], neg_arr[:n_recipes],
        pos_n_arr[:n_recipes], neg_n_arr[:n_recipes],
        stat_min_arr[:n_recipes], stat_max_arr[:n_recipes],
        stat_present_arr[:n_recipes],
        arrays.is_req_global,
        survivor_indices_buf,
    )

    n_dura_culled += (n_recipes - n_surv)

    # Build dicts only for survivors.
    out: List[dict] = []
    for i in range(n_surv):
        ridx = int(survivor_indices_buf[i])
        grid = grid_per_recipe[ridx]
        eff = eff_per_recipe[ridx]
        keys, kc, tmin_snap = keys_in_order_per_recipe[ridx]
        # Use the stored snapshot (tmin_snap) and look up tmax via stat_max_arr (full scatter).
        tmax_dense = stat_max_arr[ridx]

        ings_list = [int(arrays.ing_ids[grid[s]]) if grid[s] != -1 else -1 for s in range(6)]
        eff_list = [int(eff[s]) for s in range(6)]
        stats_dict = {}
        for kk in range(kc):
            k = int(keys[kk])
            name = arrays.stat_names[k]
            stats_dict[name] = {"min": int(tmin_snap[k]), "max": int(tmax_dense[k])}

        out.append({"ings": ings_list, "eff": eff_list, "stats": stats_dict})

    return out, n_generated, n_dura_culled


# ---------------------------------------------------------------------------
# Per-(profession, n) job
# ---------------------------------------------------------------------------

def _compute_min_dura_for_profession(recipes_data, profession, lvl_min, lvl_max):
    dura_str = 'duration' if profession in CONSU_SKILLS else 'durability'
    max_dura = 0
    for r in recipes_data:
        rd = r.data
        if (rd['skill'] == profession
                and rd['lvl']['minimum'] == lvl_min
                and rd['lvl']['maximum'] == lvl_max):
            d = rd[dura_str]['maximum']
            if d > max_dura:
                max_dura = d
    if max_dura == 0:
        return None
    return -max_dura


def _dump_line(recipe: dict) -> bytes:
    """Serialize one recipe as JSON bytes (no trailing comma/newline).

    Uses json.dumps default separators ("(', ', ': ')") so the byte stream
    matches the legacy precalc.py output exactly. Set
    PRECALC_FAST_USE_ORJSON=1 to opt into the faster compact format.
    """
    if _USE_ORJSON:
        return orjson.dumps(recipe)
    else:
        return json.dumps(recipe).encode('utf-8')


def _run_one(profession: str,
             n: int,
             ingred_path: str,
             precalc_dir: str,
             include_dura: bool,
             min_dura,
             quiet: bool = False) -> Tuple[str, int, int, int, float]:
    """Generate one META_n.json file for one profession. Returns stats."""

    t0 = time.time()
    dura_str = 'duration' if profession in CONSU_SKILLS else 'durability'

    with open(ingred_path, 'r', encoding='utf-8') as f:
        all_ingreds = json.load(f)

    arrays = _filter_and_pack(all_ingreds, profession, include_dura, dura_str)
    F = arrays.F
    G = arrays.num_global_stats

    skill_dir = os.path.join(precalc_dir, profession)
    os.makedirs(skill_dir, exist_ok=True)
    out_path = os.path.join(skill_dir, f"META_{n}.json")

    if F == 0 or G == 0:
        # Empty profession; still produce an empty list.
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('[\n]')
        return out_path, 0, 0, 0, time.time() - t0

    dura_global_idx = -1
    for k, name in enumerate(arrays.stat_names):
        if name == dura_str:
            dura_global_idx = k
            break

    # Allocate scratch buffers sized for the worst possible local batch:
    # max grids per multiset = (n! / multiset multiplicities) * C(6, n) ≤ n! * C(6, n).
    max_perms = math.factorial(n)
    max_combos = math.comb(6, n)
    max_local = max_perms * max_combos

    grid_buf = np.zeros(6, dtype=np.int32)
    eff_buf = np.zeros(6, dtype=np.int32)
    keys_buf = np.zeros(G, dtype=np.int32)
    total_min_buf = np.zeros(G, dtype=np.int32)
    total_max_buf = np.zeros(G, dtype=np.int32)
    seen_buf = np.zeros(G, dtype=np.bool_)

    pos_arr = np.zeros((max_local, 6), dtype=np.int32)
    neg_arr = np.zeros((max_local, 6), dtype=np.int32)
    pos_n_arr = np.zeros(max_local, dtype=np.int32)
    neg_n_arr = np.zeros(max_local, dtype=np.int32)
    stat_min_arr = np.zeros((max_local, G), dtype=np.int32)
    stat_max_arr = np.zeros((max_local, G), dtype=np.int32)
    stat_present_arr = np.zeros((max_local, G), dtype=np.bool_)
    survivor_indices_buf = np.zeros(max_local, dtype=np.int32)

    should_check_dura = (min_dura is not None) and (dura_global_idx >= 0)

    num_multisets = math.comb(F + n - 1, n)
    checkpoint = max(1, num_multisets // 100)

    total_generated = 0
    total_culled = 0
    total_saved = 0

    # Open output file and stream as we go. LF line endings are smaller and
    # interpret correctly across platforms.
    line_sep = b'\n'
    with open(out_path, 'wb') as fh:
        fh.write(b'[' + line_sep)
        first = True
        for ms_idx, multiset in enumerate(
            itertools.combinations_with_replacement(range(F), n)
        ):
            survivors, gen, culled = _process_multiset(
                multiset, n, arrays,
                should_check_dura, dura_global_idx, min_dura,
                grid_buf, eff_buf, keys_buf, total_min_buf, total_max_buf, seen_buf,
                pos_arr, neg_arr, pos_n_arr, neg_n_arr,
                stat_min_arr, stat_max_arr, stat_present_arr,
                survivor_indices_buf,
            )
            total_generated += gen
            total_culled += culled
            total_saved += len(survivors)

            for rec in survivors:
                if not first:
                    fh.write(b',' + line_sep)
                fh.write(_dump_line(rec))
                first = False

            if not quiet and ((ms_idx + 1) % checkpoint == 0 or (ms_idx + 1) == num_multisets):
                pct = ((ms_idx + 1) * 100) // num_multisets
                elapsed = time.time() - t0
                print(
                    f"[{profession}/META_{n}] {pct}% | gen {total_generated:,} "
                    f"saved {total_saved:,} culled {total_culled:,} "
                    f"({elapsed:.0f}s)"
                )

        fh.write(line_sep + b']')

    return out_path, total_generated, total_saved, total_culled, time.time() - t0


# ---------------------------------------------------------------------------
# Public entry: run a single profession (sequential over n=1..pre_filled).
# ---------------------------------------------------------------------------

def run_precalculation_fast(
    profession: str,
    pre_filled: int = 5,
    include_dura: bool = True,
    lvl_min: int = 103,
    lvl_max: int = 105,
    ingred_path: str = "data/ingreds_compress.json",
    recipes_path: str = "data/recipes_compress.json",
    precalc_dir: str = "data/precalc/full",
    quiet: bool = False,
) -> List[Tuple[str, int, int, int, float]]:
    os.makedirs(precalc_dir, exist_ok=True)

    recipes_data = load_recipes(recipes_path)
    min_dura = _compute_min_dura_for_profession(
        recipes_data, profession, lvl_min, lvl_max
    )

    results = []
    for n in range(1, pre_filled + 1):
        result = _run_one(
            profession, n,
            ingred_path=ingred_path,
            precalc_dir=precalc_dir,
            include_dura=include_dura,
            min_dura=min_dura,
            quiet=quiet,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Multiprocess entry point
# ---------------------------------------------------------------------------

def _job(args):
    profession, n, ingred_path, recipes_path, precalc_dir, include_dura, lvl_min, lvl_max = args
    recipes_data = load_recipes(recipes_path)
    min_dura = _compute_min_dura_for_profession(
        recipes_data, profession, lvl_min, lvl_max
    )
    return _run_one(
        profession, n,
        ingred_path=ingred_path,
        precalc_dir=precalc_dir,
        include_dura=include_dura,
        min_dura=min_dura,
        quiet=True,
    )


def run_all(
    professions: List[str],
    pre_filled: int = 5,
    include_dura: bool = True,
    lvl_min: int = 103,
    lvl_max: int = 105,
    ingred_path: str = "data/ingreds_compress.json",
    recipes_path: str = "data/recipes_compress.json",
    precalc_dir: str = "data/precalc/full",
    workers: int = None,
):
    os.makedirs(precalc_dir, exist_ok=True)
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    jobs = []
    for prof in professions:
        for n in range(1, pre_filled + 1):
            jobs.append(
                (prof, n, ingred_path, recipes_path, precalc_dir,
                 include_dura, lvl_min, lvl_max)
            )

    print(f"[precalc_fast] {len(jobs)} jobs across {workers} workers")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for result in pool.imap_unordered(_job, jobs):
            out_path, gen, saved, culled, elapsed = result
            base = os.path.basename(out_path)
            print(
                f"  done {base}: gen {gen:,} saved {saved:,} culled {culled:,} "
                f"({elapsed:.0f}s)"
            )
    print(f"[precalc_fast] all done in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    # Default to the full 8 professions if invoked standalone.
    DEFAULT_PROFESSIONS = [
        'ARMOURING', 'TAILORING', 'WEAPONSMITHING', 'WOODWORKING',
        'JEWELING', 'ALCHEMISM', 'SCRIBING', 'COOKING',
    ]
    run_all(DEFAULT_PROFESSIONS)
