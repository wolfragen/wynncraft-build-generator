"""
Numba evaluator for DSL axis programs.

Two entry points:

    eval_program(op_codes, op_args, consts, stats, ctx, scratch)
        -> float64
        Forward evaluation. Used at search leaves on a single fully-
        determined ingredient combination.

    eval_program_bounds(
        op_codes, op_args, consts,
        stats_lb, stats_ub, ctx, scratch_a, scratch_b, is_monotone
    ) -> (lb, ub)
        Computes a [LB, UB] envelope of the program's value over the
        hyper-rectangle [stats_lb, stats_ub]. Used during DFS branch-
        and-bound to prune subtrees that can't improve any axis.

For globally-monotone programs (`is_monotone=True`), the LB/UB call
runs the evaluator twice — once with each leaf reading stat_lb or
stat_ub depending on its baked corner sign, once with the inverted
choice — yielding tight bounds in O(N_ops). For non-monotone
programs the fallback evaluates at the two extreme corners
(all-LB and all-UB) and returns (min, max); this is conservative
but always valid.

Layout reminder
---------------
    op_codes : int32[N]
    op_args  : int32[N, 2]   — args0, args1
    consts   : float64[K]
    stats[*] : float64[S]    — projected stat space
    ctx      : float64[CTX_SIZE]
    scratch  : float64[max_stack]   — caller-allocated, reused
"""

import numpy as np
from numba import njit

from data.skillpoint_lookup import SKP_HEADLINE_PCT, SKP_ELEMENT_PCT, SKP_MAX
from . import opcodes as op


# ============================================================
# Forward evaluation (single corner)
# ============================================================

@njit(cache=True, fastmath=True)
def eval_program(op_codes, op_args, consts, stats, ctx, scratch,
                 skp_headline, skp_element):
    """Evaluate a program at a fully-determined stat vector.

    `skp_headline` and `skp_element` are the per-skp lookup tables —
    passed in explicitly so they're typed as numpy arrays in numba (the
    module-level constants from `skillpoint_lookup` would import as
    Python objects through @njit otherwise). Shape (5, SKP_MAX+1)."""
    sp = 0
    n = op_codes.shape[0]
    for i in range(n):
        c = op_codes[i]
        a0 = op_args[i, 0]

        if c == op.OP_LIT:
            scratch[sp] = consts[a0]
            sp += 1

        elif c == op.OP_LOAD_STAT:
            scratch[sp] = stats[a0]
            sp += 1

        elif c == op.OP_LOAD_CTX:
            scratch[sp] = ctx[a0]
            sp += 1

        elif c == op.OP_ADD:
            arity = a0
            sp -= arity
            s = 0.0
            for k in range(arity):
                s += scratch[sp + k]
            scratch[sp] = s
            sp += 1

        elif c == op.OP_MUL:
            arity = a0
            sp -= arity
            p = 1.0
            for k in range(arity):
                p *= scratch[sp + k]
            scratch[sp] = p
            sp += 1

        elif c == op.OP_SUB:
            # Stack [.., a, b]; pops b then a; result = a - b.
            sp -= 2
            scratch[sp] = scratch[sp] - scratch[sp + 1]
            sp += 1

        elif c == op.OP_DIV:
            sp -= 2
            scratch[sp] = scratch[sp] / scratch[sp + 1]
            sp += 1

        elif c == op.OP_NEG:
            scratch[sp - 1] = -scratch[sp - 1]

        elif c == op.OP_MAX:
            arity = a0
            sp -= arity
            m = scratch[sp]
            for k in range(1, arity):
                v = scratch[sp + k]
                if v > m:
                    m = v
            scratch[sp] = m
            sp += 1

        elif c == op.OP_MIN:
            arity = a0
            sp -= arity
            m = scratch[sp]
            for k in range(1, arity):
                v = scratch[sp + k]
                if v < m:
                    m = v
            scratch[sp] = m
            sp += 1

        elif c == op.OP_CLAMP:
            # Stack [.., val, lo, hi]; pop hi, lo, val.
            sp -= 3
            val = scratch[sp]
            lo = scratch[sp + 1]
            hi = scratch[sp + 2]
            if val < lo:
                val = lo
            if val > hi:
                val = hi
            scratch[sp] = val
            sp += 1

        elif c == op.OP_SP_HEADLINE:
            sp -= 1
            count = scratch[sp]
            if count < 0.0:
                count = 0.0
            elif count > SKP_MAX:
                count = SKP_MAX
            ic = int(count)
            scratch[sp] = skp_headline[a0, ic]
            sp += 1

        elif c == op.OP_SP_ELEMENT:
            sp -= 1
            count = scratch[sp]
            if count < 0.0:
                count = 0.0
            elif count > SKP_MAX:
                count = SKP_MAX
            ic = int(count)
            scratch[sp] = skp_element[a0, ic]
            sp += 1

    return scratch[0]


# ============================================================
# LB/UB evaluation
# ============================================================

@njit(cache=True, fastmath=True)
def _eval_corner(op_codes, op_args, consts, stats_lb, stats_ub, ctx,
                 scratch, want_lb, skp_headline, skp_element):
    """Evaluate the program at one corner of the [stats_lb, stats_ub]
    box. `want_lb=True` picks the LB corner: each LOAD_STAT reads
    stats_lb if its baked sign in op_args[i, 1] is +1, else stats_ub.
    `want_lb=False` picks the UB corner (signs inverted)."""
    sp = 0
    n = op_codes.shape[0]
    for i in range(n):
        c = op_codes[i]
        a0 = op_args[i, 0]
        a1 = op_args[i, 1]

        if c == op.OP_LIT:
            scratch[sp] = consts[a0]
            sp += 1

        elif c == op.OP_LOAD_STAT:
            # a1 = +1 → use lb for LB-corner, ub for UB-corner.
            # a1 = -1 → use ub for LB-corner, lb for UB-corner.
            if want_lb:
                if a1 > 0:
                    scratch[sp] = stats_lb[a0]
                else:
                    scratch[sp] = stats_ub[a0]
            else:
                if a1 > 0:
                    scratch[sp] = stats_ub[a0]
                else:
                    scratch[sp] = stats_lb[a0]
            sp += 1

        elif c == op.OP_LOAD_CTX:
            scratch[sp] = ctx[a0]
            sp += 1

        elif c == op.OP_ADD:
            arity = a0
            sp -= arity
            s = 0.0
            for k in range(arity):
                s += scratch[sp + k]
            scratch[sp] = s
            sp += 1

        elif c == op.OP_MUL:
            arity = a0
            sp -= arity
            p = 1.0
            for k in range(arity):
                p *= scratch[sp + k]
            scratch[sp] = p
            sp += 1

        elif c == op.OP_SUB:
            sp -= 2
            scratch[sp] = scratch[sp] - scratch[sp + 1]
            sp += 1

        elif c == op.OP_DIV:
            sp -= 2
            scratch[sp] = scratch[sp] / scratch[sp + 1]
            sp += 1

        elif c == op.OP_NEG:
            scratch[sp - 1] = -scratch[sp - 1]

        elif c == op.OP_MAX:
            arity = a0
            sp -= arity
            m = scratch[sp]
            for k in range(1, arity):
                v = scratch[sp + k]
                if v > m:
                    m = v
            scratch[sp] = m
            sp += 1

        elif c == op.OP_MIN:
            arity = a0
            sp -= arity
            m = scratch[sp]
            for k in range(1, arity):
                v = scratch[sp + k]
                if v < m:
                    m = v
            scratch[sp] = m
            sp += 1

        elif c == op.OP_CLAMP:
            sp -= 3
            val = scratch[sp]
            lo = scratch[sp + 1]
            hi = scratch[sp + 2]
            if val < lo:
                val = lo
            if val > hi:
                val = hi
            scratch[sp] = val
            sp += 1

        elif c == op.OP_SP_HEADLINE:
            sp -= 1
            count = scratch[sp]
            if count < 0.0:
                count = 0.0
            elif count > SKP_MAX:
                count = SKP_MAX
            ic = int(count)
            scratch[sp] = skp_headline[a0, ic]
            sp += 1

        elif c == op.OP_SP_ELEMENT:
            sp -= 1
            count = scratch[sp]
            if count < 0.0:
                count = 0.0
            elif count > SKP_MAX:
                count = SKP_MAX
            ic = int(count)
            scratch[sp] = skp_element[a0, ic]
            sp += 1

    return scratch[0]


@njit(cache=True, fastmath=True)
def eval_program_bounds(op_codes, op_args, consts, stats_lb, stats_ub, ctx,
                        scratch_a, scratch_b, is_monotone,
                        skp_headline, skp_element):
    """Returns (lb, ub) for the program over [stats_lb, stats_ub].

    Monotone programs use one corner per bound (tight, two evaluator
    passes). Non-monotone programs evaluate at the extreme corners
    (all-stats-at-lb and all-stats-at-ub) and return their min/max
    — conservative but valid.
    """
    if is_monotone:
        lb = _eval_corner(op_codes, op_args, consts, stats_lb, stats_ub, ctx,
                          scratch_a, True, skp_headline, skp_element)
        ub = _eval_corner(op_codes, op_args, consts, stats_lb, stats_ub, ctx,
                          scratch_b, False, skp_headline, skp_element)
        return lb, ub
    # Non-monotone fallback: evaluate at the two extreme corners. We
    # ignore the per-leaf sign baked at compile time (it's meaningless
    # for non-monotone programs) and force every LOAD_STAT to read
    # from a single source per pass, by passing the same array as both
    # lb and ub to _eval_corner.
    v_lo = _eval_corner(op_codes, op_args, consts, stats_lb, stats_lb, ctx,
                        scratch_a, True, skp_headline, skp_element)
    v_hi = _eval_corner(op_codes, op_args, consts, stats_ub, stats_ub, ctx,
                        scratch_b, True, skp_headline, skp_element)
    if v_lo <= v_hi:
        return v_lo, v_hi
    return v_hi, v_lo


# ============================================================
# Convenience: ensure the SKP tables stay typed for numba
# ============================================================
# Module-level constants for callers that don't want to import directly
# from skillpoint_lookup.

SKP_HEADLINE_PCT_F64 = SKP_HEADLINE_PCT.astype(np.float64, copy=False)
SKP_ELEMENT_PCT_F64 = SKP_ELEMENT_PCT.astype(np.float64, copy=False)
