import json
import os
import sys
import time
import numpy as np
from numba import njit, prange

REQ_STATS = {"strReq", "dexReq", "intReq", "defReq", "agiReq"}
DURA_STATS = {"durability", "duration"}

# Stat category codes (must stay in sync with compare_vectors).
CAT_INVERTIBLE = 0   # higher OR further-from-zero is better (invertible via -eff)
CAT_REQ        = 1   # lower is better (negative req is best)
CAT_DURA       = 2   # strictly higher is better, no sign-mismatch incomparability


# ---------------------------------------------------------
# Numba JIT Compiled Functions
# ---------------------------------------------------------
@njit(fastmath=True)
def compare_vectors(a, b, num_effs, stat_categories, dura_i):
    """
    Returns:
       1 if A dominates B
      -1 if B dominates A
       2 if A and B are identical
       0 if A and B are incomparable

    Column layout:
        [0,        num_effs)             : void-slot effectiveness (sorted desc)
        [num_effs, num_effs + num_stats) : stat representatives

    `stat_categories[s]` is one of CAT_INVERTIBLE / CAT_REQ / CAT_DURA.
    `dura_i` is the absolute column of the dura representative (or -1).
    """
    a_better = False
    b_better = False

    for i in range(len(a)):
        va = a[i]
        vb = b[i]

        if va == vb:
            continue

        # Dura/duration is non-invertible: strictly "higher is better",
        # and a positive dura genuinely dominates a negative one (no incomparability).
        if i == dura_i:
            if va > vb: a_better = True
            else:       b_better = True
            if a_better and b_better:
                return 0
            continue

        if i < num_effs:
            # ---- VOID-SLOT EFFECTIVENESS (unchanged invertibility logic) ----
            # 1. Zero is strictly worse than any non-zero.
            if va == 0:
                b_better = True
            elif vb == 0:
                a_better = True
            # 2. Positive and negative effs are incomparable.
            elif (va > 0 and vb < 0) or (va < 0 and vb > 0):
                return 0
            # 3. Both positive: higher is better.
            elif va > 0:
                if va > vb: a_better = True
                else:       b_better = True
            # 4. Both negative: further from zero is better (more invertible).
            else:
                if va < vb: a_better = True
                else:       b_better = True
        else:
            # ---- STAT (req or invertible) ----
            # Strict sign opposition is incomparable for invertible/req stats.
            if (va > 0 and vb < 0) or (va < 0 and vb > 0):
                return 0

            is_req = stat_categories[i - num_effs] == CAT_REQ

            if va > 0 or vb > 0:
                # Both >= 0 with at least one > 0.
                # "!= is_req" flips the direction for requirements (lower is better).
                if (va > vb) != is_req:
                    a_better = True
                else:
                    b_better = True
            else:
                # Both <= 0.
                # For non-req invertible stats, "0 vs -X" is incomparable:
                # the 0 side has no stat at all (always 0), while -X is only
                # useful via -eff inversion for one direction but not the other.
                # Treat like a sign mismatch.
                if not is_req and (va == 0 or vb == 0):
                    return 0
                # Otherwise: further-below-zero = lower req for reqs, and
                # bigger magnitude for invertible stats with both strictly < 0.
                if va < vb: a_better = True
                else:       b_better = True

        if a_better and b_better:
            return 0

    if a_better: return 1
    if b_better: return -1
    return 2  # Identical


@njit(cache=True)
def pareto_filter(matrix, num_effs, stat_categories, dura_i):
    """Reference single-thread Pareto cull. Kept for validation/diffing."""
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    for i in range(n):
        if not is_kept[i]:
            continue

        for j in range(i + 1, n):
            if not is_kept[j]:
                continue

            cmp = compare_vectors(matrix[i], matrix[j], num_effs, stat_categories, dura_i)

            if cmp == 1 or cmp == 2:
                is_kept[j] = False
            elif cmp == -1:
                is_kept[i] = False
                break

    return is_kept


# ---------------------------------------------------------
# Block-parallel Pareto cull
# ---------------------------------------------------------
# Produces a VALID Pareto set (no survivor dominates another survivor), but the
# specific set may differ in rare cases from the single-thread reference because
# the comparator is non-transitive (pos-req vs neg-req is incomparable). The user
# has accepted this trade-off: the final output after downstream steps is
# order-independent for the use case.

@njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _parallel_check_block(
    block_start,
    block_end,
    survivors,
    num_survivors,
    matrix,
    num_effs,
    stat_categories,
    dura_i,
    is_kept,
):
    """
    Phase 1 — for each candidate c in [block_start, block_end), test against all
    `survivors` (indices from prior blocks, known alive at block start, ascending).

    Parallelism is across candidates (prange). Writes:
      - is_kept[c]   : thread-owned, safe.
      - is_kept[s]   : shared, but only monotonically flipped False. Bool writes
                       are byte-atomic on x86; idempotent across threads.

    We do NOT read is_kept[s] inside the inner loop — that would create
    non-deterministic output based on thread interleaving. Always iterate all
    survivors.
    """
    count = block_end - block_start
    for local in prange(count):
        c = block_start + local
        for k in range(num_survivors):
            s = survivors[k]
            cmp = compare_vectors(
                matrix[s], matrix[c], num_effs, stat_categories, dura_i
            )
            if cmp == 1 or cmp == 2:
                is_kept[c] = False
                break
            elif cmp == -1:
                is_kept[s] = False


@njit(nogil=True, fastmath=True, cache=True)
def _intra_block_cull(
    block_start,
    block_end,
    matrix,
    num_effs,
    stat_categories,
    dura_i,
    is_kept,
):
    """Phase 2 — sequential cull among the candidates of a single block."""
    for i in range(block_start, block_end):
        if not is_kept[i]:
            continue
        for j in range(i + 1, block_end):
            if not is_kept[j]:
                continue
            cmp = compare_vectors(
                matrix[i], matrix[j], num_effs, stat_categories, dura_i
            )
            if cmp == 1 or cmp == 2:
                is_kept[j] = False
            elif cmp == -1:
                is_kept[i] = False
                break


def pareto_filter_block(matrix, num_effs, stat_categories, dura_i, block_size=1024):
    """
    Block-parallel Pareto cull.

    Maintains an ascending-index list of survivors from prior blocks. For each
    new block: (1) parallel scan of candidates against survivors, (2) single-
    thread intra-block cull, (3) merge alive block items into the survivor list.
    """
    n = matrix.shape[0]
    is_kept = np.ones(n, dtype=np.bool_)

    # `survivors` is a sorted int64 array of alive indices < block_start.
    survivors = np.empty(0, dtype=np.int64)

    block_start = 0
    while block_start < n:
        block_end = min(block_start + block_size, n)

        if survivors.size > 0:
            _parallel_check_block(
                block_start,
                block_end,
                survivors,
                survivors.size,
                matrix,
                num_effs,
                stat_categories,
                dura_i,
                is_kept,
            )

        _intra_block_cull(
            block_start,
            block_end,
            matrix,
            num_effs,
            stat_categories,
            dura_i,
            is_kept,
        )

        # Prune survivors killed during the parallel phase, append new survivors.
        if survivors.size > 0:
            survivors = survivors[is_kept[survivors]]
        block_indices = np.arange(block_start, block_end, dtype=np.int64)
        alive_in_block = block_indices[is_kept[block_start:block_end]]
        if alive_in_block.size > 0:
            survivors = np.concatenate((survivors, alive_in_block))

        block_start = block_end

    return is_kept


# ---------------------------------------------------------
# Main Python Wrapper
# ---------------------------------------------------------
def cull_recipes(
    skill,
    n,
    in_dir=os.path.join("data", "precalc", "full"),
    out_dir=os.path.join("data", "precalc", "generic_cull"),
    block_size=1024,
    use_parallel=True,
):
    filename = f"META_{n}.json"
    in_path = os.path.join(in_dir, skill, filename)
    out_path = os.path.join(out_dir, skill, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    print("Pass 1/3: Reading and mapping stats...")
    stat_keys = set()
    raw_lines = []
    num_effs = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().rstrip(',')
            if clean in ('[', ']', ''):
                continue
            raw_lines.append(clean)

            try:
                cand = json.loads(clean)
                if "stats" in cand:
                    stat_keys.update(cand["stats"].keys())
                if num_effs == 0:
                    ings = cand.get("ingredients", cand.get("ings", []))
                    effs = cand.get("effectiveness", cand.get("eff", []))
                    num_effs = len([e for i, e in zip(ings, effs) if i == -1])
            except:
                continue

    if not raw_lines:
        return

    stat_list = sorted(stat_keys)
    stat_to_idx = {k: i for i, k in enumerate(stat_list)}
    num_stats = len(stat_list)
    num_lines = len(raw_lines)

    # Per-stat category vector + absolute dura column index.
    stat_categories = np.full(num_stats, CAT_INVERTIBLE, dtype=np.int32)
    dura_i = -1
    for s_name, s_idx in stat_to_idx.items():
        if s_name in REQ_STATS:
            stat_categories[s_idx] = CAT_REQ
        elif s_name in DURA_STATS:
            stat_categories[s_idx] = CAT_DURA
            dura_i = num_effs + s_idx

    print(f"Pass 2/3: Building {num_lines}x{num_effs + num_stats} matrix...")
    matrix = np.zeros((num_lines, num_effs + num_stats), dtype=np.float32)

    for row_idx, clean in enumerate(raw_lines):
        cand = json.loads(clean)
        ings = cand.get("ingredients", cand.get("ings", []))
        effs_list = cand.get("effectiveness", cand.get("eff", []))

        # Void effs sorted descending so column-wise comparison lines up across recipes.
        effs = [float(str(e).replace('%', '')) for i, e in zip(ings, effs_list) if i == -1]
        effs.sort(reverse=True)
        for i, val in enumerate(effs):
            matrix[row_idx, i] = val

        if "stats" in cand:
            for s_name, d_val in cand["stats"].items():
                s_idx = stat_to_idx[s_name]
                col_idx = num_effs + s_idx
                d_min = d_val["min"]
                d_max = d_val["max"]
                cat = stat_categories[s_idx]
                # Representative selection mirrors the runtime numba_cull:
                #   reqs           -> min if max > 0 else max  (binding side for "lower is better")
                #   invertible/dura -> max if max > 0 else min (best upside / most-invertible downside)
                if cat == CAT_REQ:
                    matrix[row_idx, col_idx] = d_min if d_max > 0 else d_max
                else:
                    matrix[row_idx, col_idx] = d_max if d_max > 0 else d_min

    print(f"Pass 3/3: Running {'block-parallel' if use_parallel else 'single-thread'} dominance cull...")

    # Sort heuristic: encounter likely-dominant recipes first so they kill the
    # weaker ones early. Score per category:
    #   - eff: |val|, since invertibility makes magnitude what counts.
    #   - req: -rep (signed) — high positive req hurts, negative req helps.
    #   - dura: +rep (signed) — strict higher-is-better.
    #   - invertible: +|rep| — strong values either direction tend to dominate.
    eff_scores = np.sum(np.abs(matrix[:, :num_effs]), axis=1)
    stat_scores = np.zeros(num_lines, dtype=np.float32)
    for s_idx in range(num_stats):
        col = num_effs + s_idx
        cat = stat_categories[s_idx]
        if cat == CAT_REQ:
            stat_scores -= matrix[:, col]
        elif cat == CAT_DURA:
            stat_scores += matrix[:, col]
        else:
            stat_scores += np.abs(matrix[:, col])
    sort_order = np.argsort(-(eff_scores + stat_scores))

    matrix = matrix[sort_order]
    raw_lines = [raw_lines[i] for i in sort_order]

    t_cull = time.time()
    if use_parallel:
        is_kept = pareto_filter_block(
            matrix, num_effs, stat_categories, dura_i, block_size=block_size
        )
    else:
        is_kept = pareto_filter(matrix, num_effs, stat_categories, dura_i)
    kept_indices = np.where(is_kept)[0]
    print(f"  cull took {time.time() - t_cull:.1f}s, kept {len(kept_indices)}/{num_lines}")

    print(f"Writing {len(kept_indices)} recipes to output...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, idx in enumerate(kept_indices):
            f.write(raw_lines[idx])
            f.write(",\n" if i < len(kept_indices) - 1 else "\n")
        f.write("]")

    print(f"Done! Saved to {out_path}")


def run_all(
    professions,
    pre_filled=5,
    in_dir=os.path.join("data", "precalc", "full"),
    out_dir=os.path.join("data", "precalc", "generic_cull"),
    block_size=1024,
    use_parallel=True,
):
    t0 = time.time()
    for profession in professions:
        for i in range(1, pre_filled + 1):
            print(f"\n=== {profession}/META_{i}.json ===")
            t_file = time.time()
            cull_recipes(
                profession,
                i,
                in_dir=in_dir,
                out_dir=out_dir,
                block_size=block_size,
                use_parallel=use_parallel,
            )
            print(f"  file total: {time.time() - t_file:.1f}s")
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    PROFESSIONS = [
        "ARMOURING",
        "TAILORING",
        "WEAPONSMITHING",
        "WOODWORKING",
        "JEWELING",
        "ALCHEMISM",
        "SCRIBING",
        "COOKING",
    ]
    run_all(
        PROFESSIONS,
        pre_filled=5,
        in_dir=os.path.join("data", "precalc", "full"),
        out_dir=os.path.join("data", "precalc", "generic_cull"),
    )
