"""
Streaming/iterative variant of precalc_culling.cull_recipes.

Instead of loading the entire input file into a single matrix (which peaks
RAM at ~7GB for the new ARMOURING/META_5 4.5GB input), this version:

  1. Pass 1 — single linear scan to discover stat keys + num_effs.
     Doesn't retain raw lines.
  2. Pass 2 — stream the file in batches of `batch_size` lines:
     a. parse + build a small (batch x ncols) matrix
     b. sort the batch by the same dominance heuristic
     c. vstack [persistent_survivors, batch] and run pareto_filter_block
        once over the combined matrix
     d. update persistent_survivors with what survived
     e. drop batch from RAM
  3. Write all surviving raw lines out.

Output is identical to precalc_culling.cull_recipes (same algorithm, just
chunked memory access). Reuses the existing JIT kernels unchanged.

Trade-off: heuristic sort is per-batch only (vs global), so a slightly
weaker "encounter dominators first" property. In practice the survivor
set still converges since dominators in later batches kill earlier ones
during vstack-cull. Marginally more work than a perfectly-sorted single
pass; vastly less RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

from precalc_culling import (
    REQ_STATS,
    DURA_STATS,
    CAT_INVERTIBLE,
    CAT_REQ,
    CAT_DURA,
    pareto_filter_block,
    _parallel_check_block,
)


def _scan_metadata(in_path: str) -> Tuple[List[str], int, int]:
    """Pass 1: discover stat_keys + num_effs without retaining lines.

    Returns (sorted_stat_keys, num_effs, total_recipe_lines).
    """
    stat_keys: set = set()
    num_effs = 0
    n_lines = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().rstrip(",")
            if clean in ("[", "]", ""):
                continue
            try:
                cand = json.loads(clean)
            except Exception:
                continue
            n_lines += 1
            if "stats" in cand:
                stat_keys.update(cand["stats"].keys())
            if num_effs == 0:
                ings = cand.get("ingredients", cand.get("ings", []))
                effs = cand.get("effectiveness", cand.get("eff", []))
                num_effs = len([e for i, e in zip(ings, effs) if i == -1])
    return sorted(stat_keys), num_effs, n_lines


def _build_batch_matrix(
    parsed_batch: list,
    stat_to_idx: dict,
    num_effs: int,
    num_stats: int,
    stat_categories: np.ndarray,
) -> np.ndarray:
    """Build (n_batch, ncols) matrix for one batch of pre-parsed recipes."""
    n = len(parsed_batch)
    matrix = np.zeros((n, num_effs + num_stats), dtype=np.float32)
    for row_idx, cand in enumerate(parsed_batch):
        ings = cand.get("ingredients", cand.get("ings", []))
        effs_list = cand.get("effectiveness", cand.get("eff", []))
        effs = [float(str(e).replace("%", "")) for i, e in zip(ings, effs_list) if i == -1]
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
                if cat == CAT_REQ:
                    matrix[row_idx, col_idx] = d_min if d_max > 0 else d_max
                else:
                    matrix[row_idx, col_idx] = d_max if d_max > 0 else d_min
    return matrix


def _heuristic_order(
    matrix: np.ndarray,
    num_effs: int,
    num_stats: int,
    stat_categories: np.ndarray,
) -> np.ndarray:
    """Same per-row dominance score as precalc_culling.cull_recipes."""
    eff_scores = np.sum(np.abs(matrix[:, :num_effs]), axis=1)
    stat_scores = np.zeros(matrix.shape[0], dtype=np.float32)
    for s_idx in range(num_stats):
        col = num_effs + s_idx
        cat = stat_categories[s_idx]
        if cat == CAT_REQ:
            stat_scores -= matrix[:, col]
        elif cat == CAT_DURA:
            stat_scores += matrix[:, col]
        else:
            stat_scores += np.abs(matrix[:, col])
    return np.argsort(-(eff_scores + stat_scores))


def cull_recipes_streaming(
    skill: str,
    n: int,
    in_dir: str = os.path.join("data", "precalc", "full"),
    out_dir: str = os.path.join("data", "precalc", "generic_cull"),
    batch_size: int = 200_000,
    block_size: int = 1024,
):
    filename = f"META_{n}.json"
    in_path = os.path.join(in_dir, skill, filename)
    out_path = os.path.join(out_dir, skill, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    file_mb = os.path.getsize(in_path) / 1024 / 1024
    print(f"Pass 1/2: scanning {file_mb:.0f} MB for stat keys + num_effs...")
    t0 = time.time()
    stat_list, num_effs, total_lines = _scan_metadata(in_path)
    if total_lines == 0:
        print("  no recipes found, skipping")
        return
    print(f"  {len(stat_list)} stats, {num_effs} void-eff slots, {total_lines:,} lines "
          f"({time.time() - t0:.1f}s)")

    stat_to_idx = {k: i for i, k in enumerate(stat_list)}
    num_stats = len(stat_list)
    stat_categories = np.full(num_stats, CAT_INVERTIBLE, dtype=np.int32)
    dura_i = -1
    for s_name, s_idx in stat_to_idx.items():
        if s_name in REQ_STATS:
            stat_categories[s_idx] = CAT_REQ
        elif s_name in DURA_STATS:
            stat_categories[s_idx] = CAT_DURA
            dura_i = num_effs + s_idx
    ncols = num_effs + num_stats

    print(f"Pass 2/2: streaming cull, batch={batch_size:,}, block={block_size:,}...")

    survivor_matrix = np.empty((0, ncols), dtype=np.float32)
    survivor_lines: List[str] = []

    raw_batch: List[str] = []
    parsed_batch: list = []
    total_processed = 0
    t_cull = time.time()

    def flush_batch():
        nonlocal survivor_matrix, survivor_lines, total_processed
        if not raw_batch:
            return
        batch_matrix = _build_batch_matrix(
            parsed_batch, stat_to_idx, num_effs, num_stats, stat_categories
        )
        # Per-batch heuristic sort (cheap, helps the cull within the batch).
        order = _heuristic_order(batch_matrix, num_effs, num_stats, stat_categories)
        batch_matrix = batch_matrix[order]
        batch_lines_sorted = [raw_batch[i] for i in order]

        # Combine survivors-so-far with the new batch and let the
        # block-parallel cull do its job. Survivors come first because they
        # were already sorted/dominated; new batch entries that beat them
        # will still kill them in the cross-block scan.
        if survivor_matrix.shape[0] > 0:
            combined = np.vstack([survivor_matrix, batch_matrix])
            combined_lines = survivor_lines + batch_lines_sorted
        else:
            combined = batch_matrix
            combined_lines = batch_lines_sorted

        is_kept = pareto_filter_block(
            combined, num_effs, stat_categories, dura_i, block_size=block_size
        )
        kept = np.where(is_kept)[0]
        survivor_matrix = combined[kept].copy()  # detach from `combined`
        survivor_lines = [combined_lines[i] for i in kept]

        total_processed += len(raw_batch)
        elapsed = time.time() - t_cull
        rate = total_processed / max(elapsed, 1e-9)
        pct = total_processed * 100.0 / total_lines
        print(
            f"  {pct:5.1f}%  processed {total_processed:>12,}  "
            f"survivors {len(kept):>10,}  "
            f"{rate:>8,.0f} l/s  elapsed {elapsed:.0f}s"
        )
        raw_batch.clear()
        parsed_batch.clear()

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().rstrip(",")
            if clean in ("[", "]", ""):
                continue
            try:
                cand = json.loads(clean)
            except Exception:
                continue
            raw_batch.append(clean)
            parsed_batch.append(cand)
            if len(raw_batch) >= batch_size:
                flush_batch()
    flush_batch()

    print(f"  cull total {time.time() - t_cull:.0f}s, final survivors {len(survivor_lines):,}")
    print(f"Writing {len(survivor_lines):,} recipes to {out_path}...")
    t_write = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("[\n")
        for i, line in enumerate(survivor_lines):
            fh.write(line)
            fh.write(",\n" if i < len(survivor_lines) - 1 else "\n")
        fh.write("]")
    print(f"Done {skill}/META_{n} (write {time.time() - t_write:.1f}s)")


def _checkpoint_path(out_path: str) -> str:
    return out_path + ".checkpoint"


def _safe_replace(src: str, dst: str, retries: int = 20, delay: float = 0.5):
    """os.replace with retry on Windows PermissionError.

    Antivirus (Avast, Defender) routinely scans newly-created files, holding
    a transient handle that blocks rename. Retry with backoff instead of
    propagating — the lock typically clears in <1s.
    """
    last_err: Exception = OSError("unreached")
    for i in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay * (1 + i * 0.5))  # mild backoff
    raise last_err


def _checkpoint_save(out_path: str, survivor_lines, total_processed, input_path):
    """Single-file atomic checkpoint.

    Format:
      line 1: JSON header {total_processed, input_mtime, input_size, n_survivors}
      lines 2..N+1: one survivor JSON per line

    Write goes to .tmp then atomic os.replace. A crash between write and replace
    leaves the previous checkpoint untouched. A crash mid-write leaves only the
    .tmp which load() ignores.
    """
    cp_path = _checkpoint_path(out_path)
    cp_tmp = cp_path + ".tmp"
    header = {
        "total_processed": total_processed,
        "input_mtime": os.path.getmtime(input_path),
        "input_size": os.path.getsize(input_path),
        "n_survivors": len(survivor_lines),
    }
    with open(cp_tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(header))
        f.write("\n")
        for line in survivor_lines:
            f.write(line)
            f.write("\n")
    _safe_replace(cp_tmp, cp_path)


def _checkpoint_load(out_path: str, input_path: str):
    """Return (survivor_lines, total_processed) or None if no valid checkpoint."""
    cp_path = _checkpoint_path(out_path)
    if not os.path.isfile(cp_path):
        return None
    try:
        with open(cp_path, "r", encoding="utf-8") as f:
            first = f.readline()
            if not first:
                return None
            try:
                header = json.loads(first.rstrip("\n"))
            except Exception:
                return None
            if (header.get("input_mtime") != os.path.getmtime(input_path)
                    or header.get("input_size") != os.path.getsize(input_path)):
                print(f"  [checkpoint] input changed since checkpoint, ignoring")
                return None
            survivor_lines: List[str] = []
            for line in f:
                line = line.rstrip("\n")
                if line:
                    survivor_lines.append(line)
        if len(survivor_lines) != header.get("n_survivors"):
            print(f"  [checkpoint] truncated (n_survivors mismatch), ignoring")
            return None
        return survivor_lines, int(header["total_processed"])
    except Exception as e:
        print(f"  [checkpoint] read error {e}, ignoring")
        return None


def _checkpoint_clean(out_path: str):
    import shutil
    cp_path = _checkpoint_path(out_path)
    for p in (cp_path, cp_path + ".tmp"):
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _checkpoint_normalize(out_path: str):
    """Heal pre-existing state. Removes a leftover checkpoint *directory* (from
    a prior version that used a dir layout) so atomic os.replace can succeed."""
    import shutil
    cp_path = _checkpoint_path(out_path)
    if os.path.isdir(cp_path):
        shutil.rmtree(cp_path, ignore_errors=True)
    cp_tmp = cp_path + ".tmp"
    if os.path.isdir(cp_tmp):
        shutil.rmtree(cp_tmp, ignore_errors=True)


def cull_recipes_streaming_v2(
    skill: str,
    n: int,
    in_dir: str = os.path.join("data", "precalc", "full"),
    out_dir: str = os.path.join("data", "precalc", "generic_cull"),
    batch_size: int = 200_000,
    block_size: int = 1024,
    checkpoint: bool = True,
):
    """Optimized streaming cull.

    Key insight: in v1 we vstack [survivors, batch] and call pareto_filter_block
    on the full combined matrix. That re-runs intra-block + cross-block compares
    on survivors that were ALREADY Pareto-optimal — so every survivor-vs-survivor
    compare is wasted (they're guaranteed incomparable, all return 0). For
    ARMOURING/META_5 with ~1M survivors, this dominates total runtime.

    v2 splits the work explicitly:
      1. Parallel intra-batch cull   -> batch_kept (no survivor checks yet)
      2. Single parallel scan: batch_kept candidates vs survivors
         (uses _parallel_check_block directly, NOT block-iterated)
      3. Update survivor list

    Avoids O(survivors^2) wasted compares per batch. Output set is identical
    to v1 (same comparator, same kernels, just orchestrated to skip redundant
    work).
    """
    filename = f"META_{n}.json"
    in_path = os.path.join(in_dir, skill, filename)
    out_path = os.path.join(out_dir, skill, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return

    file_mb = os.path.getsize(in_path) / 1024 / 1024
    print(f"Pass 1/2: scanning {file_mb:.0f} MB for stat keys + num_effs...")
    t0 = time.time()
    stat_list, num_effs, total_lines = _scan_metadata(in_path)
    if total_lines == 0:
        print("  no recipes found, skipping")
        return
    print(f"  {len(stat_list)} stats, {num_effs} void-eff slots, {total_lines:,} lines "
          f"({time.time() - t0:.1f}s)")

    stat_to_idx = {k: i for i, k in enumerate(stat_list)}
    num_stats = len(stat_list)
    stat_categories = np.full(num_stats, CAT_INVERTIBLE, dtype=np.int32)
    dura_i = -1
    for s_name, s_idx in stat_to_idx.items():
        if s_name in REQ_STATS:
            stat_categories[s_idx] = CAT_REQ
        elif s_name in DURA_STATS:
            stat_categories[s_idx] = CAT_DURA
            dura_i = num_effs + s_idx
    ncols = num_effs + num_stats

    survivor_matrix = np.empty((0, ncols), dtype=np.float32)
    survivor_lines: List[str] = []
    skip_lines = 0  # number of input recipe lines to skip (resume from checkpoint)

    if checkpoint:
        _checkpoint_normalize(out_path)
        loaded = _checkpoint_load(out_path, in_path)
        if loaded is not None:
            survivor_lines, skip_lines = loaded
            # Re-parse survivors and rebuild matrix.
            t_resume = time.time()
            parsed_resume = []
            for s in survivor_lines:
                try:
                    parsed_resume.append(json.loads(s))
                except Exception:
                    pass
            survivor_matrix = _build_batch_matrix(
                parsed_resume, stat_to_idx, num_effs, num_stats, stat_categories
            )
            print(f"  [checkpoint] resumed: {len(survivor_lines):,} survivors, "
                  f"skipping {skip_lines:,} input lines "
                  f"(rebuild {time.time() - t_resume:.1f}s)")

    print(f"Pass 2/2 (v2): streaming cull, batch={batch_size:,}, block={block_size:,}, "
          f"checkpoint={'on' if checkpoint else 'off'}")

    raw_batch: List[str] = []
    parsed_batch: list = []
    total_processed = skip_lines
    t_cull = time.time()

    def flush_batch():
        nonlocal survivor_matrix, survivor_lines, total_processed
        if not raw_batch:
            return

        # Build + sort batch by heuristic.
        batch_matrix = _build_batch_matrix(
            parsed_batch, stat_to_idx, num_effs, num_stats, stat_categories
        )
        order = _heuristic_order(batch_matrix, num_effs, num_stats, stat_categories)
        batch_matrix = batch_matrix[order]
        batch_lines_sorted = [raw_batch[i] for i in order]

        # Step 1: parallel intra-batch Pareto cull.
        batch_kept_mask = pareto_filter_block(
            batch_matrix, num_effs, stat_categories, dura_i, block_size=block_size
        )
        kept_idx = np.where(batch_kept_mask)[0]
        if kept_idx.size > 0:
            batch_matrix = batch_matrix[kept_idx]
            batch_lines_kept = [batch_lines_sorted[i] for i in kept_idx]

            if survivor_matrix.shape[0] == 0:
                survivor_matrix = batch_matrix.copy()
                survivor_lines = batch_lines_kept
            else:
                # Step 2: cross-cull batch_kept vs survivors. ONE parallel scan.
                S = survivor_matrix.shape[0]
                K = batch_matrix.shape[0]
                combined = np.vstack([survivor_matrix, batch_matrix])
                is_kept = np.ones(S + K, dtype=np.bool_)
                survivors_idx = np.arange(S, dtype=np.int64)
                _parallel_check_block(
                    S, S + K,
                    survivors_idx, S,
                    combined,
                    num_effs, stat_categories, dura_i,
                    is_kept,
                )
                new_idx = np.where(is_kept)[0]
                survivor_matrix = combined[new_idx]
                combined_lines = survivor_lines + batch_lines_kept
                survivor_lines = [combined_lines[i] for i in new_idx]

        total_processed += len(raw_batch)
        elapsed = time.time() - t_cull
        rate = (total_processed - skip_lines) / max(elapsed, 1e-9)
        pct = total_processed * 100.0 / total_lines

        # Persist checkpoint BEFORE clearing batch buffers — if we crash mid-write,
        # the next run still resumes from prior batch (last good checkpoint).
        if checkpoint:
            t_cp = time.time()
            _checkpoint_save(out_path, survivor_lines, total_processed, in_path)
            cp_dt = time.time() - t_cp
            cp_str = f" cp {cp_dt:.1f}s"
        else:
            cp_str = ""

        print(
            f"  {pct:5.1f}%  processed {total_processed:>12,}  "
            f"survivors {len(survivor_lines):>10,}  "
            f"{rate:>8,.0f} l/s  elapsed {elapsed:.0f}s{cp_str}"
        )
        raw_batch.clear()
        parsed_batch.clear()

    line_idx = 0  # count of valid recipe lines seen
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip().rstrip(",")
            if clean in ("[", "]", ""):
                continue
            try:
                cand = json.loads(clean)
            except Exception:
                continue
            line_idx += 1
            if line_idx <= skip_lines:
                continue
            raw_batch.append(clean)
            parsed_batch.append(cand)
            if len(raw_batch) >= batch_size:
                flush_batch()
    flush_batch()

    print(f"  cull total {time.time() - t_cull:.0f}s, final survivors {len(survivor_lines):,}")
    print(f"Writing {len(survivor_lines):,} recipes to {out_path}...")
    t_write = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("[\n")
        for i, line in enumerate(survivor_lines):
            fh.write(line)
            fh.write(",\n" if i < len(survivor_lines) - 1 else "\n")
        fh.write("]")
    if checkpoint:
        _checkpoint_clean(out_path)
    print(f"Done {skill}/META_{n} (write {time.time() - t_write:.1f}s)")


def run_all(
    professions: List[str],
    pre_filled: int = 5,
    in_dir: str = os.path.join("data", "precalc", "full"),
    out_dir: str = os.path.join("data", "precalc", "generic_cull"),
    batch_size: int = 200_000,
    block_size: int = 1024,
    skip_existing: bool = False,
    use_v2: bool = True,
):
    t0 = time.time()
    for prof in professions:
        for i in range(1, pre_filled + 1):
            out_path = os.path.join(out_dir, prof, f"META_{i}.json")
            in_path = os.path.join(in_dir, prof, f"META_{i}.json")
            if (
                skip_existing
                and os.path.exists(out_path)
                and os.path.exists(in_path)
                and os.path.getmtime(out_path) >= os.path.getmtime(in_path)
            ):
                print(f"\n=== {prof}/META_{i}.json (skip — already fresh) ===")
                continue
            print(f"\n=== {prof}/META_{i}.json ===")
            t_file = time.time()
            fn = cull_recipes_streaming_v2 if use_v2 else cull_recipes_streaming
            fn(
                prof, i,
                in_dir=in_dir, out_dir=out_dir,
                batch_size=batch_size, block_size=block_size,
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
    # Allow CLI override: precalc_culling_iter.py [skip_existing] [batch_size]
    skip_existing = "--skip-existing" in sys.argv
    batch_size = 200_000
    for arg in sys.argv[1:]:
        if arg.startswith("--batch="):
            batch_size = int(arg.split("=", 1)[1])
    run_all(
        PROFESSIONS,
        pre_filled=5,
        batch_size=batch_size,
        skip_existing=skip_existing,
    )
