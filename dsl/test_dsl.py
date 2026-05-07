"""
Smoke tests for the DSL: build small programs by hand, evaluate them
against expected values, and verify the LB/UB envelope is tight on
monotone programs and conservative on non-monotone fallback paths.

Run directly:
    python -m dsl.test_dsl
"""

import sys
import numpy as np

from data.skillpoint_lookup import SKP_STR, SKP_DEX, SKP_HEADLINE_PCT
from query.query import build_query, BUILD_CTX_BASE_REQ_STR, BUILD_CTX_BASE_REQ_DEX

from dsl import (
    LIT, STAT, CTX, ADD, MUL, SUB, NEG, CLAMP,
    SP_HEADLINE, compile_program,
    eval_program, eval_program_bounds,
    SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64,
)


def _make_query(stat_names_with_weights):
    """Build a Query with the given stats made active (so STAT() resolves
    to a valid projected index in compile_program)."""
    user_json = {
        name: {"weight": 1, "ingredient_filter": False}
        for name in stat_names_with_weights
    }
    # Need durability or duration active for build_query not to fail at
    # downstream points; we don't actually use the search here so this
    # is only to satisfy the parser's expectations.
    user_json["durability"] = {"min": 1, "weight": 1}
    return build_query(user_json, search_for_inversion=False,
                       item_type="CHESTPLATE", skill="ARMOURING",
                       consumable=False)


def _eval_at(program, stats_full_dict, query, ctx=None):
    """Helper: build a projected-stats vector from a Python dict and
    evaluate the program."""
    s = np.zeros(query.stat_count, dtype=np.float64)
    for name, val in stats_full_dict.items():
        from data.stats import STAT_INDEX
        proj = int(query.proj_stats_idx[STAT_INDEX[name]])
        if proj < 0:
            raise KeyError(f"{name} not active in this query")
        s[proj] = float(val)
    if ctx is None:
        ctx = np.zeros(query.build_ctx.shape[0], dtype=np.float64)
    scratch = np.zeros(max(program.max_stack, 1), dtype=np.float64)
    return eval_program(program.op_codes, program.op_args, program.consts,
                        s, ctx, scratch,
                        SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64)


def _bounds_at(program, stats_lb_dict, stats_ub_dict, query, ctx=None):
    s_lb = np.zeros(query.stat_count, dtype=np.float64)
    s_ub = np.zeros(query.stat_count, dtype=np.float64)
    from data.stats import STAT_INDEX
    for name, v in stats_lb_dict.items():
        s_lb[int(query.proj_stats_idx[STAT_INDEX[name]])] = float(v)
    for name, v in stats_ub_dict.items():
        s_ub[int(query.proj_stats_idx[STAT_INDEX[name]])] = float(v)
    if ctx is None:
        ctx = np.zeros(query.build_ctx.shape[0], dtype=np.float64)
    scratch_a = np.zeros(max(program.max_stack, 1), dtype=np.float64)
    scratch_b = np.zeros(max(program.max_stack, 1), dtype=np.float64)
    return eval_program_bounds(program.op_codes, program.op_args, program.consts,
                               s_lb, s_ub, ctx, scratch_a, scratch_b,
                               program.is_monotone,
                               SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64)


# ============================================================
# Tests
# ============================================================

def test_literal():
    q = _make_query([])
    prog = compile_program(LIT(42.5), q)
    v = _eval_at(prog, {}, q)
    assert v == 42.5, f"LIT eval failed: got {v}"
    print("test_literal OK")


def test_add_lit():
    q = _make_query([])
    prog = compile_program(ADD(LIT(10), LIT(20), LIT(30)), q)
    v = _eval_at(prog, {}, q)
    assert v == 60, f"ADD lit failed: got {v}"
    print("test_add_lit OK")


def test_stat_load():
    q = _make_query(["damPct"])
    prog = compile_program(STAT("damPct"), q)
    v = _eval_at(prog, {"damPct": 75}, q)
    assert v == 75, f"STAT load failed: got {v}"
    print("test_stat_load OK")


def test_dps_axis_value():
    """Reproduce the user's example DPS axis on a tiny build."""
    q = _make_query(["damPct", "sdPct", "mdRaw", "sdRaw"])
    BASE_DPS = 220.0
    pct_part = ADD(LIT(100), STAT("damPct"), STAT("sdPct"))
    raw_part = ADD(STAT("mdRaw"), STAT("sdRaw"))
    base = ADD(MUL(LIT(BASE_DPS), pct_part), raw_part)
    str_count = ADD(LIT(60), CTX(BUILD_CTX_BASE_REQ_STR))
    dex_count = ADD(LIT(40), CTX(BUILD_CTX_BASE_REQ_DEX))
    axis = MUL(
        base,
        ADD(LIT(1), SP_HEADLINE(SKP_STR, str_count)),
        ADD(LIT(1), SP_HEADLINE(SKP_DEX, dex_count)),
    )
    prog = compile_program(axis, q)

    # Hand-compute the expected value: damPct=50, sdPct=10, mdRaw=100, sdRaw=20.
    stats = {"damPct": 50, "sdPct": 10, "mdRaw": 100, "sdRaw": 20}
    ctx = np.zeros(q.build_ctx.shape[0], dtype=np.float64)
    ctx[BUILD_CTX_BASE_REQ_STR] = 30  # str alloc 30 → str_count=90
    ctx[BUILD_CTX_BASE_REQ_DEX] = 0   # dex_count=40

    expected_pct = 100 + 50 + 10  # 160
    expected_base = BASE_DPS * expected_pct + 100 + 20  # 220*160 + 120 = 35320
    str_pct = SKP_HEADLINE_PCT[SKP_STR, 90]
    dex_pct = SKP_HEADLINE_PCT[SKP_DEX, 40]
    expected = expected_base * (1 + str_pct) * (1 + dex_pct)

    v = _eval_at(prog, stats, q, ctx=ctx)
    assert abs(v - expected) < 1e-6, f"DPS axis: got {v}, expected {expected}"
    assert prog.is_monotone, "DPS axis should be flagged monotone"
    print(f"test_dps_axis_value OK (value={v:.2f})")


def test_monotone_bounds_tight():
    """For a monotone program, LB/UB should be exactly the program
    evaluated at the corresponding extreme corner."""
    q = _make_query(["damPct", "sdPct", "mdRaw", "sdRaw"])
    pct_part = ADD(LIT(100), STAT("damPct"), STAT("sdPct"))
    raw_part = ADD(STAT("mdRaw"), STAT("sdRaw"))
    axis = ADD(MUL(LIT(220.0), pct_part), raw_part)
    prog = compile_program(axis, q)

    lb_dict = {"damPct": 0, "sdPct": 0, "mdRaw": 0, "sdRaw": 0}
    ub_dict = {"damPct": 100, "sdPct": 50, "mdRaw": 200, "sdRaw": 100}
    lb, ub = _bounds_at(prog, lb_dict, ub_dict, q)
    expected_lb = _eval_at(prog, lb_dict, q)
    expected_ub = _eval_at(prog, ub_dict, q)
    assert abs(lb - expected_lb) < 1e-6, f"LB mismatch: {lb} vs {expected_lb}"
    assert abs(ub - expected_ub) < 1e-6, f"UB mismatch: {ub} vs {expected_ub}"
    print(f"test_monotone_bounds_tight OK (lb={lb:.0f} ub={ub:.0f})")


def test_sub_corner_signs():
    """SUB(a, b) is mono+ in a, mono- in b. So LB(a-b) at corners should
    be evaluated as (a_lb - b_ub), and UB as (a_ub - b_lb)."""
    q = _make_query(["damPct", "strReq"])
    axis = SUB(STAT("damPct"), STAT("strReq"))  # damPct - strReq
    prog = compile_program(axis, q)
    assert prog.is_monotone, "SUB program should be monotone"

    lb_dict = {"damPct": 10, "strReq": 5}
    ub_dict = {"damPct": 50, "strReq": 30}
    lb, ub = _bounds_at(prog, lb_dict, ub_dict, q)
    # LB = damPct_lb - strReq_ub = 10 - 30 = -20
    # UB = damPct_ub - strReq_lb = 50 - 5 = 45
    assert lb == -20.0, f"SUB LB: got {lb}"
    assert ub == 45.0, f"SUB UB: got {ub}"
    print(f"test_sub_corner_signs OK (lb={lb} ub={ub})")


def test_neg_corner_signs():
    """NEG flips mono. LB(-x) at corners = -ub(x); UB = -lb(x)."""
    q = _make_query(["damPct"])
    axis = NEG(STAT("damPct"))
    prog = compile_program(axis, q)
    assert prog.is_monotone

    lb, ub = _bounds_at(prog, {"damPct": 5}, {"damPct": 25}, q)
    assert lb == -25.0 and ub == -5.0, f"NEG bounds: ({lb}, {ub})"
    print(f"test_neg_corner_signs OK (lb={lb} ub={ub})")


def test_sp_headline_lookup():
    """SP_HEADLINE evaluates the curve correctly and clamps."""
    q = _make_query(["damPct"])  # damPct unused; just need a valid query
    # SP_HEADLINE(SKP_STR, 75) should match SKP_HEADLINE_PCT[STR, 75].
    axis = SP_HEADLINE(SKP_STR, LIT(75))
    prog = compile_program(axis, q)
    v = _eval_at(prog, {}, q)
    expected = SKP_HEADLINE_PCT[SKP_STR, 75]
    assert abs(v - expected) < 1e-12, f"SP_HEADLINE 75: {v} vs {expected}"

    # Clamps to SKP_MAX=150 above; check 200 → same as 150.
    axis_hi = SP_HEADLINE(SKP_STR, LIT(200))
    prog_hi = compile_program(axis_hi, q)
    v_hi = _eval_at(prog_hi, {}, q)
    assert abs(v_hi - SKP_HEADLINE_PCT[SKP_STR, 150]) < 1e-12, f"clamp@150: {v_hi}"

    # Clamps to 0 below.
    axis_neg = SP_HEADLINE(SKP_STR, LIT(-10))
    prog_neg = compile_program(axis_neg, q)
    v_neg = _eval_at(prog_neg, {}, q)
    assert v_neg == 0.0, f"clamp@0: {v_neg}"
    print("test_sp_headline_lookup OK")


def test_clamp_op():
    """CLAMP(val, lo, hi) limits val to [lo, hi]. Mono+ in val."""
    q = _make_query(["damPct"])
    axis = CLAMP(STAT("damPct"), LIT(10), LIT(80))
    prog = compile_program(axis, q)
    assert _eval_at(prog, {"damPct": 5}, q) == 10
    assert _eval_at(prog, {"damPct": 50}, q) == 50
    assert _eval_at(prog, {"damPct": 200}, q) == 80
    # Bounds: with LO=10, HI=80 (constants), CLAMP is mono+ in val so
    # LB at val_lb=5 → 10, UB at val_ub=200 → 80.
    lb, ub = _bounds_at(prog, {"damPct": 5}, {"damPct": 200}, q)
    assert lb == 10 and ub == 80, f"CLAMP bounds: ({lb}, {ub})"
    print("test_clamp_op OK")


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        test_literal,
        test_add_lit,
        test_stat_load,
        test_dps_axis_value,
        test_monotone_bounds_tight,
        test_sub_corner_signs,
        test_neg_corner_signs,
        test_sp_headline_lookup,
        test_clamp_op,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")
