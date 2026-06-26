"""
milp/result.py

Validate a MILP pick with the craft.js-faithful oracle (no new scoring code, no
URL round-trip — call craft_core directly), and pretty-print the report.

Two consistency checks turn a "proven optimal" into a trustworthy result:
  1. The b-tensor / model effectiveness must equal the oracle's effectiveness for
     the chosen build (else the model's E[p] is wrong — a bug, not a fidelity gap).
  2. The pick must be oracle-VALID against the hard min/max (else the model's
     constraints don't match the query — also a bug). Distinct from the expected
     small numeric drift between the MILP's linear objective and the floored oracle
     score (milp/README.md §5).
"""

from craft_core import compute_effectiveness, compute_crafted_stats, score_query


def verify_milp_pick(ingredient_ids, recipe_raw, tier, ing_by_id, user_query,
                     mm, res, atk_speed="NORMAL"):
    """ingredient_ids: 6 JSON ids in slot order. Returns a report dict."""
    entries = [ing_by_id[int(i)] for i in ingredient_ids]
    eff_flat = list(compute_effectiveness(entries))

    chosen = res["chosen_rows"]
    # (1a) effectiveness reconstructed from the b-tensor over the chosen rows.
    E_recon = [100 + int(sum(mm.b[q, chosen[q], p] for q in range(6)))
               for p in range(6)]
    # (1b) the model's own E[p] decision-variable values.
    solver = res["solver"]
    E_model = [int(solver.Value(mm.E[p])) for p in range(6)]

    eff_match = (E_recon == eff_flat)
    model_eff_match = (E_model == eff_flat)

    crafted = compute_crafted_stats(recipe_raw.data, tier, entries, eff_flat,
                                    atk_speed=atk_speed)
    scored = score_query(crafted, user_query)

    return {
        "eff_flat": eff_flat,
        "E_recon": E_recon,
        "E_model": E_model,
        "eff_match": eff_match,
        "model_eff_match": model_eff_match,
        "crafted": crafted,
        "oracle_score": scored["score"],
        "valid": scored["valid"],
        "violations": scored["violations"],
        "notes": scored["notes"],
    }


def print_report(res, ingredient_ids, url, report, ing_by_id, solve_t=None):
    print("=" * 64)
    print(f"MILP status      : {res['status_name']}", end="")
    if res["status_name"] == "OPTIMAL":
        print("   (mathematically proven optimal for the v1 linear model)")
    elif res["status_name"] == "FEASIBLE":
        print("   (best found within time limit - NOT proven optimal)")
    else:
        print()
    if solve_t is not None:
        print(f"Solve time       : {solve_t:.2f}s")
    if res["objective"] is not None:
        print(f"MILP objective   : {res['objective']:.0f}  "
              f"(integer-scaled; bound {res['best_bound']:.0f})")

    names = [ing_by_id[int(i)].get("name", "?") for i in ingredient_ids]
    print("\nSlots (0..5):")
    for p, (jid, nm) in enumerate(zip(ingredient_ids, names)):
        print(f"  slot {p}: id={jid:<5} eff={report['eff_flat'][p]:>5}%  {nm}")

    print(f"\nEffectiveness    : {report['eff_flat']}")
    print(f"  b-tensor recon : {report['E_recon']}  "
          f"{'MATCH' if report['eff_match'] else 'MISMATCH!'}")
    print(f"  model E[p] vars: {report['E_model']}  "
          f"{'MATCH' if report['model_eff_match'] else 'MISMATCH!'}")

    verdict = "VALID" if report["valid"] else "INVALID"
    print(f"\nOracle score     : {report['oracle_score']:.2f}   [{verdict}]")
    if report["violations"]:
        print("Constraint violations (model/query mismatch — investigate):")
        for v in report["violations"]:
            print(f"  - {v}")
    if report["notes"]:
        print("Notes:")
        for n in report["notes"]:
            print(f"  - {n}")

    print(f"\nCrafter URL      : {url}")
    print("\nNote: v1 MILP drops per-stat floor() rounding and SP->% non-linear "
          "scoring (milp/README.md s5); small drift vs the oracle score is "
          "expected. Effectiveness and inversion are exact.")
    print("=" * 64)
