# Developer Documentation — Wynncraft Build Generator

This folder is the onboarding map for the `python/` project. It explains **how the
crafted-item optimiser works**, module by module, so a new contributor (human or
AI) can find the relevant code fast and understand the non-obvious conventions
before changing anything.

> The top-level [`../README.md`](../README.md) is the user-facing summary and the
> authoritative list of **known soundness caveats**. This folder is the deeper,
> code-level companion. When the two disagree about *behaviour*, trust the code
> and fix the doc.

## Start here

1. **[01-architecture.md](01-architecture.md)** — the big picture: what a "craft"
   is, the meta/normal ingredient split, the end-to-end data flow through
   `main.py`, and where each subsystem fits.
2. **[08-glossary-and-gotchas.md](08-glossary-and-gotchas.md)** — vocabulary
   (eff, posMods, META_n, void slot, inversion, composite…) and the consolidated
   list of conventions that bite. Skim this early; come back to it often.

## Per-subsystem references

| Doc | Covers | Source files |
|-----|--------|--------------|
| [02-data-layer.md](02-data-layer.md) | Stat registry, ingredient/recipe loading, the compact search DB, spells & skillpoint curves, on-disk JSON formats | `data/stats.py`, `data/ingredient_loader.py`, `data/ingredient_db.py`, `data/recipe.py`, `data/recipe_loader.py`, `data/spells.py`, `data/skillpoint_lookup.py` |
| [03-query.md](03-query.md) | Parsing the user query into dense arrays, the projected stat space, composites, `lower_better` direction logic | `query/query.py` |
| [04-search-engine.md](04-search-engine.md) | The branch-and-bound DFS, pruning bounds, useful masks, composite bounding, numba/warmup, pipelined load+search | `core/search_engine.py`, `core/warmup.py` |
| [05-precalc-and-meta-sets.md](05-precalc-and-meta-sets.md) | Offline enumeration of meta arrangements, Pareto culling, the runtime meta-set loader + binary cache | `precalc_fast.py`, `precalc_culling.py`, `precalc_culling_iter.py`, `data/meta_set_loader.py`, `query/ingredient_filter.py` |
| [06-pareto-mode.md](06-pareto-mode.md) | The alternative K-axis Pareto-frontier search and its DSL | `core/pareto_search.py`, `dsl/*`, `main_pareto.py` |
| [07-encoding-and-tools.md](07-encoding-and-tools.md) | Crafter-URL bit encoding, the decode/score tool, the entrypoints | `utils/hash_generator.py`, `main_decode.py`, `main.py`, `main_build_temp.py` |

## Conventions used in these docs

- **`file.py:NN`** points at a line. Line numbers drift; treat them as a strong
  hint, not gospel. The surrounding function name is the durable anchor.
- **"Ground truth"** means the reference JS implementation in
  `../wynnbuilder.github.io-master/` (especially `crafter/craft.js`). The Python
  port deliberately mirrors its arithmetic; when in doubt, that JS is canonical.
- `python - Copie/` (sibling of `python/`) is a **stale backup** — ignore it.

## Repo layout note

Only `python/` is a git repository; the parent working directory is not (the
environment banner that says "git repo: false" refers to the parent). See the
session memory note `reference_git_layout`.
