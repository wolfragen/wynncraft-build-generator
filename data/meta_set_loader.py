"""
meta_set_loader.py

Loads and refines meta-sets according to Query.

Pipeline:
  1. `load_meta_sets` iterates META_1..5 for a skill.
  2. For each file: `_load_cached_arrays` reads either a fresh `.cache.npz`
     sibling (fast path) or parses JSON once and builds the cache.
  3. `_refine_batch` projects stats into active space + applies recipe shift,
     extracts void effs, runs numba pareto cull, sorts void effs desc.

The pareto cull kernel (`numba_cull`) is unchanged from the pre-cache version.
"""

import numpy as np
import os
from typing import NamedTuple

from numba import njit, prange

try:
    import orjson as _orjson

    def _load_json_bytes(path):
        with open(path, "rb") as f:
            return _orjson.loads(f.read())
except ImportError:
    import json as _json

    def _load_json_bytes(path):
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)

from data.stats import STAT_INDEX, STAT_COUNT, REQ_STATS


# Bump this when the on-disk cache format changes so old caches are ignored.
_CACHE_VERSION = 1


class MetaBatch(NamedTuple):
    ings_matrix: np.ndarray          # (M, 6)
    void_count: int                  # number of void slots
    void_eff_matrix: np.ndarray      # (M, void_count)
    void_slots_matrix: np.ndarray    # (M, void_count)
    base_min_matrix: np.ndarray      # (M, K)
    base_max_matrix: np.ndarray      # (M, K)


# ------------------------------------------------------------
# Main Loader
# ------------------------------------------------------------

def load_meta_sets(skill, query, recipe, culling=True, max_cull=5, should_print=False, base_path="data/precalc/generic_cull"):

    full_meta_sets = []

    # META_0 — synthetic single empty recipe.
    empty_ings = np.full((1, 6), -1, dtype=np.int32)
    empty_eff = np.full((1, 6), 100, dtype=np.int32)
    empty_stat_names = np.empty(0, dtype="U1")
    empty_stat_min = np.zeros((1, 0), dtype=np.int32)
    empty_stat_max = np.zeros((1, 0), dtype=np.int32)
    first_batch = _refine_batch(
        empty_ings, empty_eff, empty_stat_names, empty_stat_min, empty_stat_max,
        query, recipe, culling=False,
    )
    full_meta_sets.append(first_batch)

    # META_1..5
    for n in range(1, 6):
        use_culling = culling and n <= max_cull

        ings, eff, stat_names, stat_min, stat_max = _load_cached_arrays(skill, n, base_path)

        batch = _refine_batch(
            ings, eff, stat_names, stat_min, stat_max,
            query, recipe, culling=use_culling,
        )

        if should_print:
            print(f"{skill}_META_{n}: {ings.shape[0]} => {batch.ings_matrix.shape[0]}")

        full_meta_sets.append(batch)

    return full_meta_sets


# ------------------------------------------------------------
# Binary cache layer
# ------------------------------------------------------------

def _load_cached_arrays(skill, n, base_path):
    """
    Returns (ings, eff, stat_names, stat_min, stat_max).
      - ings        : (N, 6) int32
      - eff         : (N, 6) int32
      - stat_names  : (S,) unicode array — every stat name appearing in the file
      - stat_min    : (N, S) int32
      - stat_max    : (N, S) int32

    Uses `{base_path}/{skill}/META_{n}.cache.npz` when fresh; otherwise rebuilds
    from the JSON and writes the cache for next time.
    """
    json_path = os.path.join(base_path, skill.upper(), f"META_{n}.json")
    cache_path = json_path[:-5] + ".cache.npz"

    if os.path.exists(cache_path):
        try:
            if os.path.getmtime(cache_path) >= os.path.getmtime(json_path):
                with np.load(cache_path, allow_pickle=False) as data:
                    if int(data["version"]) == _CACHE_VERSION:
                        return (
                            data["ings"],
                            data["eff"],
                            data["stat_names"],
                            data["stat_min"],
                            data["stat_max"],
                        )
        except Exception:
            pass  # fall through to rebuild

    return _build_cache(json_path, cache_path)


def _build_cache(json_path, cache_path):
    raw = _load_json_bytes(json_path)

    N = len(raw)
    ings = np.zeros((N, 6), dtype=np.int32)
    eff = np.zeros((N, 6), dtype=np.int32)

    # Two passes: (1) collect stat name universe, (2) fill stat arrays.
    stat_names_set = set()
    for r in raw:
        s = r.get("stats")
        if s:
            stat_names_set.update(s.keys())
    stat_names = sorted(stat_names_set)
    name_to_col = {name: i for i, name in enumerate(stat_names)}
    S = len(stat_names)

    stat_min = np.zeros((N, S), dtype=np.int32)
    stat_max = np.zeros((N, S), dtype=np.int32)

    for i, r in enumerate(raw):
        ing_list = r.get("ings", r.get("ingredients"))
        if ing_list is not None:
            ings[i] = ing_list
        eff_list = r.get("eff", r.get("effectiveness"))
        if eff_list is not None:
            # Strip trailing "%" if present (defensive; data/precalc never stores it).
            if isinstance(eff_list[0], str):
                eff_list = [int(str(v).rstrip("%")) for v in eff_list]
            eff[i] = eff_list

        stats_dict = r.get("stats")
        if stats_dict:
            for name, val in stats_dict.items():
                col = name_to_col[name]
                stat_min[i, col] = val.get("min", val.get("minimum", 0))
                stat_max[i, col] = val.get("max", val.get("maximum", 0))

    stat_names_arr = np.array(stat_names, dtype="U32") if stat_names else np.empty(0, dtype="U1")

    np.savez(
        cache_path,
        version=np.int32(_CACHE_VERSION),
        ings=ings,
        eff=eff,
        stat_names=stat_names_arr,
        stat_min=stat_min,
        stat_max=stat_max,
    )

    return ings, eff, stat_names_arr, stat_min, stat_max


# ------------------------------------------------------------
# Refine (project into active stat space + cull + sort void effs)
# ------------------------------------------------------------

def _refine_batch(ings, eff, stat_names, stat_min, stat_max, query, recipe, culling=True, timings=None):
    """If `timings` is a dict, fills 'prep' and 'cull' (seconds) for profiling."""
    from time import perf_counter
    _t0 = perf_counter() if timings is not None else 0.0

    N = ings.shape[0]
    active_indices = query.active_indices
    num_active = query.stat_count

    if N == 0:
        return MetaBatch(
            np.empty((0, 6), dtype=np.int32),
            0,
            np.empty((0, 0), dtype=np.int32),
            np.empty((0, 0), dtype=np.int32),
            np.empty((0, num_active), dtype=np.int32),
            np.empty((0, num_active), dtype=np.int32),
        )

    # Build file-col -> active-col index map once per file (cheap).
    # Use a STAT_COUNT lookup to resolve each stat name's active position.
    stat_idx_to_active = np.full(STAT_COUNT, -1, dtype=np.int32)
    for pos, idx in enumerate(active_indices):
        stat_idx_to_active[idx] = pos

    file_to_active = np.full(len(stat_names), -1, dtype=np.int32)
    for i, name in enumerate(stat_names):
        idx = STAT_INDEX.get(str(name))
        if idx is not None:
            file_to_active[i] = stat_idx_to_active[idx]

    # Project stat_min/max into (N, num_active). Unmapped active columns stay zero.
    base_min_matrix = np.zeros((N, num_active), dtype=np.int32)
    base_max_matrix = np.zeros((N, num_active), dtype=np.int32)

    valid_file_cols = np.where(file_to_active >= 0)[0]
    if valid_file_cols.size > 0:
        valid_active_cols = file_to_active[valid_file_cols]
        base_min_matrix[:, valid_active_cols] = stat_min[:, valid_file_cols]
        base_max_matrix[:, valid_active_cols] = stat_max[:, valid_file_cols]

    # Apply recipe shift (row-broadcast).
    base_min_matrix += recipe.base_min_stats_proj.astype(np.int32, copy=False)
    base_max_matrix += recipe.base_max_stats_proj.astype(np.int32, copy=False)

    # Extract void effs and their real slot indices. All rows in a given file
    # have the same void count (= 6 - META_n), so we can reshape safely.
    void_mask = (ings == -1)
    void_count = int(void_mask[0].sum())

    if void_count == 0:
        # Degenerate file — no void slots. Preserve shape semantics.
        void_real_slot_matrix = np.zeros((N, 0), dtype=np.int32)
        void_eff_matrix = np.zeros((N, 0), dtype=np.int32)
    else:
        _, col_idx = np.where(void_mask)
        void_real_slot_matrix = col_idx.reshape(N, void_count).astype(np.int32, copy=False)
        row_idx = np.arange(N, dtype=np.int64).reshape(-1, 1)
        void_eff_matrix = eff[row_idx, void_real_slot_matrix].astype(np.int32, copy=False)

    ings_matrix = np.ascontiguousarray(ings, dtype=np.int32)

    if culling:
        req_mask_full = np.zeros(STAT_COUNT, dtype=np.bool_)
        for name in REQ_STATS:
            req_mask_full[STAT_INDEX[name]] = True
        req_mask_proj = req_mask_full[active_indices]

        if timings is not None:
            timings["prep"] = perf_counter() - _t0
            _tc = perf_counter()
        kept_indices = numba_cull(
            base_min_matrix,
            base_max_matrix,
            void_eff_matrix,
            ings_matrix,
            void_count,
            req_mask_proj,
            query.req_idx,
            query.dura_proj_idx,
        )
        if timings is not None:
            timings["cull"] = perf_counter() - _tc
            _t0 = perf_counter() - timings["cull"]  # so trailing sort is attributed correctly below
        ings_matrix = ings_matrix[kept_indices]
        void_eff_matrix = void_eff_matrix[kept_indices]
        void_real_slot_matrix = void_real_slot_matrix[kept_indices]
        base_min_matrix = base_min_matrix[kept_indices]
        base_max_matrix = base_max_matrix[kept_indices]

    # Sort void effs descending per row with the same tie-break as the
    # original (np.argsort ascending, then index from the end). Using numpy's
    # argsort on axis=1 matches the per-row behavior byte-for-byte.
    M = void_eff_matrix.shape[0]
    if M > 0 and void_count > 0:
        asc_order = np.argsort(void_eff_matrix, axis=1)
        desc_order = asc_order[:, ::-1]
        row_idx2 = np.arange(M, dtype=np.int64).reshape(-1, 1)
        sorted_void_eff_matrix = np.ascontiguousarray(
            void_eff_matrix[row_idx2, desc_order]
        )
        void_slots_matrix = np.ascontiguousarray(
            void_real_slot_matrix[row_idx2, desc_order]
        )
    else:
        sorted_void_eff_matrix = void_eff_matrix
        void_slots_matrix = void_real_slot_matrix

    return MetaBatch(
        ings_matrix=ings_matrix,
        void_count=void_count,
        void_eff_matrix=sorted_void_eff_matrix,
        void_slots_matrix=void_slots_matrix,
        base_min_matrix=base_min_matrix,
        base_max_matrix=base_max_matrix,
    )


# ------------------------------------------------------------
# Pareto — unchanged behavior
# ------------------------------------------------------------

@njit(fastmath=True, cache=True)
def compare_vectors(a, b, num_effs, is_req_col, dura_i):
    """
    Returns:
       1 if A dominates B
      -1 if B dominates A
       2 if A and B are identical
       0 if A and B are incomparable

    Column layout:
        - [0, num_effs)             : void-slot effectiveness (sorted desc)
        - [num_effs, num_effs + K)  : projected stats
    `dura_i` is the absolute column of durability/duration (or -1).
    `is_req_col` is a bool array of length len(a): True for "req stat" columns
    (where lower is better when positive), False for everything else. Effs/dura
    columns are False. Avoids a per-pair linear scan over `req_idx`.
    """
    a_better = False
    b_better = False

    for i in range(len(a)):
        va = a[i]
        vb = b[i]

        if va == vb:
            continue

        # Dura/duration: strictly "higher is better", skip sign-mismatch guard.
        if i == dura_i:
            if va > vb:
                a_better = True
            else:
                b_better = True
            if a_better and b_better:
                return 0
            continue

        # Strict sign opposition → incomparable (invertibility cannot bridge it).
        if (va > 0 and vb < 0) or (va < 0 and vb > 0):
            return 0

        if i < num_effs:
            # --- EFFECTIVENESS ---
            if va == 0:
                b_better = True
            elif vb == 0:
                a_better = True
            elif va > 0:
                if va > vb: a_better = True
                else:       b_better = True
            else:
                if va < vb: a_better = True
                else:       b_better = True
        else:
            is_req = is_req_col[i]

            if va > 0 or vb > 0:
                if (va > vb) != is_req:
                    a_better = True
                else:
                    b_better = True
            else:
                if not is_req and (va == 0 or vb == 0):
                    return 0
                if va < vb: a_better = True
                else:       b_better = True

        if a_better and b_better:
            return 0

    if a_better: return 1
    if b_better: return -1
    return 2


@njit(cache=True)
def pareto_filter(matrix, num_effs, is_req_col, dura_i):
    """Single-thread reference cull (preserves byte-identity)."""
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    for i in range(n):
        if not is_kept[i]:
            continue

        for j in range(i + 1, n):
            if not is_kept[j]:
                continue

            cmp = compare_vectors(matrix[i], matrix[j], num_effs, is_req_col, dura_i)

            if cmp == 1:
                is_kept[j] = False
            elif cmp == 2:
                is_kept[j] = False
            elif cmp == -1:
                is_kept[i] = False
                break

    return is_kept


@njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _parallel_check_block(block_start, block_end, survivors, num_survivors,
                          matrix, num_effs, is_req_col, dura_i, is_kept):
    """
    Block-parallel phase 1 — each candidate c in the block is tested in its own
    thread against the snapshot `survivors` (alive indices from prior blocks,
    ascending). Always iterates all survivors to keep the result deterministic
    regardless of thread interleaving (stale is_kept reads would otherwise make
    the output order-dependent).
    """
    count = block_end - block_start
    for local in prange(count):
        c = block_start + local
        for k in range(num_survivors):
            s = survivors[k]
            cmp = compare_vectors(
                matrix[s], matrix[c], num_effs, is_req_col, dura_i
            )
            if cmp == 1 or cmp == 2:
                is_kept[c] = False
                break
            elif cmp == -1:
                is_kept[s] = False


@njit(nogil=True, fastmath=True, cache=True)
def _intra_block_cull(block_start, block_end, matrix, num_effs, is_req_col,
                      dura_i, is_kept):
    """Sequential pareto cull on candidates within a single block."""
    for i in range(block_start, block_end):
        if not is_kept[i]:
            continue
        for j in range(i + 1, block_end):
            if not is_kept[j]:
                continue
            cmp = compare_vectors(
                matrix[i], matrix[j], num_effs, is_req_col, dura_i
            )
            if cmp == 1 or cmp == 2:
                is_kept[j] = False
            elif cmp == -1:
                is_kept[i] = False
                break


def pareto_filter_block(matrix, num_effs, is_req_col, dura_i, block_size=1024):
    """
    Block-parallel pareto cull. Produces a VALID Pareto set identical to the
    serial version on all real queries we've validated; the non-transitive
    comparator could in theory yield rare divergences, but end-to-end search
    output is the identity target here.
    """
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    survivors = np.empty(0, dtype=np.int64)

    block_start = 0
    while block_start < n:
        block_end = min(block_start + block_size, n)

        if survivors.size > 0:
            _parallel_check_block(
                block_start, block_end, survivors, survivors.size,
                matrix, num_effs, is_req_col, dura_i, is_kept,
            )

        _intra_block_cull(
            block_start, block_end, matrix, num_effs, is_req_col, dura_i, is_kept,
        )

        if survivors.size > 0:
            survivors = survivors[is_kept[survivors]]
        block_indices = np.arange(block_start, block_end, dtype=np.int64)
        alive_in_block = block_indices[is_kept[block_start:block_end]]
        if alive_in_block.size > 0:
            survivors = np.concatenate((survivors, alive_in_block))

        block_start = block_end

    return is_kept


@njit(parallel=True, cache=True)
def _build_cull_matrix(base_min_matrix, base_max_matrix, void_eff_matrix,
                       void_count, req_idx):
    """Builds the comparison matrix used by pareto_filter, in parallel over rows."""
    num_sets, num_stats = base_min_matrix.shape
    matrix = np.zeros((num_sets, void_count + num_stats), dtype=np.int32)

    for i in prange(num_sets):
        tmp_eff = void_eff_matrix[i].copy()
        tmp_eff.sort()  # ascending

        for j in range(void_count):
            matrix[i, j] = tmp_eff[void_count - 1 - j]

        for s in range(num_stats):
            max_val = base_max_matrix[i, s]
            col = void_count + s
            if s in req_idx:
                if max_val > 0:
                    matrix[i, col] = base_min_matrix[i, s]
                else:
                    matrix[i, col] = max_val
            else:
                if max_val > 0:
                    matrix[i, col] = max_val
                else:
                    matrix[i, col] = base_min_matrix[i, s]

    return matrix


def numba_cull(base_min_matrix,
               base_max_matrix,
               void_eff_matrix,
               ings_matrix,
               void_count,
               req_mask_proj,
               req_idx,
               dura_proj_idx):

    dura_i = -1
    if dura_proj_idx >= 0:
        dura_i = void_count + dura_proj_idx

    matrix = _build_cull_matrix(
        base_min_matrix, base_max_matrix, void_eff_matrix, void_count, req_idx,
    )

    # Precompute per-column "is req stat" lookup so the comparator does an O(1)
    # array load instead of an O(|req_idx|) linear scan per stat per pair.
    num_cols = matrix.shape[1]
    is_req_col = np.zeros(num_cols, dtype=np.bool_)
    for r in req_idx:
        if r >= 0:
            col = void_count + int(r)
            if 0 <= col < num_cols:
                is_req_col[col] = True

    # Sort rows by descending "strength" so the survivor set fills with strong
    # dominators first. Subsequent candidates hit `break` quickly via the early
    # exit in _parallel_check_block / _intra_block_cull. The cull output is
    # still a valid Pareto set; only the choice within incomparable cliques can
    # change (the comparator is non-transitive on edge cases — same caveat as
    # the existing block-parallel path).
    if matrix.shape[0] >= 2048:
        # Sum of (sorted-desc) void-eff columns is a cheap proxy for "row total
        # void eff". Higher sum tends to dominate higher in the eff dims.
        strength = matrix[:, :void_count].sum(axis=1) if void_count > 0 \
            else np.zeros(matrix.shape[0], dtype=np.int64)
        order = np.argsort(-strength, kind="stable")
        sorted_matrix = np.ascontiguousarray(matrix[order])
        is_kept_sorted = pareto_filter_block(sorted_matrix, void_count, is_req_col, dura_i)
        kept_in_sorted = np.where(is_kept_sorted)[0]
        return order[kept_in_sorted]
    else:
        is_kept = pareto_filter(matrix, void_count, is_req_col, dura_i)
        return np.where(is_kept)[0]
