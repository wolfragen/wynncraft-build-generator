"""
DSL AST builder + compiler.

Users construct axis programs via the helper functions exposed at module
level (LIT, STAT, CTX, ADD, MUL, ...). Each call returns an `Expr` AST
node; the resulting tree is passed to `compile_program(expr, query)` to
produce the flat numba-friendly arrays the evaluator consumes.

Example
-------
    from dsl.builder import LIT, STAT, ADD, MUL, SP_MULT, compile_program
    from data.skillpoint_lookup import SKP_STR, SKP_DEX
    from query.query import BUILD_CTX_BASE_REQ_STR

    # axis = (BASE_DPS * (100 + damPct + sdPct) + damRaw + sdRaw)
    #        * (1 + headline_pct(60 + base_str_alloc))
    #        * (1 + headline_pct(40 + base_dex_alloc))
    BASE_DPS = LIT(220.0)
    pct_part = ADD(LIT(100), STAT('damPct'), STAT('sdPct'))
    raw_part = ADD(STAT('damRaw'), STAT('sdRaw'))
    base = ADD(MUL(BASE_DPS, pct_part), raw_part)

    str_count = ADD(LIT(60), CTX(BUILD_CTX_BASE_REQ_STR))
    dex_count = ADD(LIT(40), CTX(BUILD_CTX_BASE_REQ_DEX))
    axis = MUL(
        base,
        ADD(LIT(1), SP_HEADLINE(SKP_STR, str_count)),
        ADD(LIT(1), SP_HEADLINE(SKP_DEX, dex_count)),
    )

    program = compile_program(axis, query)
    # program.op_codes, program.op_args, program.consts ready to feed numba.

Notes
-----
- Stat references resolve at compile time via `query.proj_stats_idx[STAT_INDEX[name]]`.
  Unknown or inactive stats raise at compile time, which is much better
  than a silent zero at runtime.
- The compiler emits postfix (RPN) — the evaluator walks linearly,
  using a small stack.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

from data.stats import STAT_INDEX
from . import opcodes as op


# ============================================================
# AST node
# ============================================================

@dataclass
class Expr:
    """One AST node. `args` may contain other Expr nodes or raw ints/floats
    (for literal-style operands like a skp_idx). Concrete shape depends on
    the opcode — the helper functions below build them correctly."""
    opcode: int
    # First int arg, e.g. skp_idx for SP_HEADLINE, arity for ADD/MUL. -1
    # if unused (literal value lives in `consts` instead).
    arg0: int = -1
    # Children (subexpressions) in argument order. For LIT, instead the
    # raw float value lives in `lit_value` and `children` is empty.
    children: Tuple["Expr", ...] = ()
    lit_value: float = 0.0
    # Resolved-at-compile-time fields (filled by the helper builders).
    stat_name: str = ""
    ctx_idx: int = -1


# ============================================================
# Builder helpers (the public API)
# ============================================================

def LIT(value: float) -> Expr:
    """Numeric constant. Can be int or float; stored as float64."""
    return Expr(opcode=op.OP_LIT, lit_value=float(value))


def STAT(name: str) -> Expr:
    """Look up a stat by its IDS_STATS / REQ_STATS / special name. The
    stat must be active in the Query at compile time (otherwise the
    compiler raises). The compiled program stores the projected index."""
    return Expr(opcode=op.OP_LOAD_STAT, stat_name=name)


def CTX(ctx_idx: int) -> Expr:
    """Read a slot of the build context array. `ctx_idx` is one of the
    BUILD_CTX_* constants from `query.query`."""
    return Expr(opcode=op.OP_LOAD_CTX, ctx_idx=int(ctx_idx))


def ADD(*args) -> Expr:
    """N-ary sum; arity must be >= 1. Numeric Python literals are
    auto-wrapped in LIT for ergonomics."""
    if not args:
        raise ValueError("ADD requires at least one operand")
    return Expr(opcode=op.OP_ADD, arg0=len(args), children=tuple(_wrap(a) for a in args))


def MUL(*args) -> Expr:
    """N-ary product. Same arity rule as ADD. ASSUMES all operands stay
    in the non-negative regime — the LB/UB evaluator declares MUL as
    monotone+ which only holds when args >= 0. If your formula can mix
    signs, structure it so MUL only multiplies non-negative subterms,
    or use NEG to factor signs explicitly."""
    if not args:
        raise ValueError("MUL requires at least one operand")
    return Expr(opcode=op.OP_MUL, arg0=len(args), children=tuple(_wrap(a) for a in args))


def SUB(a, b) -> Expr:
    """Binary subtraction: a - b. Pushes a then b in compiled order; the
    evaluator pops (b, a) and computes a - b."""
    return Expr(opcode=op.OP_SUB, children=(_wrap(a), _wrap(b)))


def DIV(a, b) -> Expr:
    """Binary division: a / b. Caller must ensure b > 0 in all reachable
    states; no runtime guard."""
    return Expr(opcode=op.OP_DIV, children=(_wrap(a), _wrap(b)))


def NEG(a) -> Expr:
    return Expr(opcode=op.OP_NEG, children=(_wrap(a),))


def MAXOP(*args) -> Expr:
    if not args:
        raise ValueError("MAX requires at least one operand")
    return Expr(opcode=op.OP_MAX, arg0=len(args), children=tuple(_wrap(a) for a in args))


def MINOP(*args) -> Expr:
    if not args:
        raise ValueError("MIN requires at least one operand")
    return Expr(opcode=op.OP_MIN, arg0=len(args), children=tuple(_wrap(a) for a in args))


def CLAMP(val, lo, hi) -> Expr:
    """clamp(val, lo, hi) = min(hi, max(lo, val)). Marked non-monotone
    in lo and hi so the LB/UB evaluator falls back to two-corner
    evaluation if either is dynamic; in practice lo/hi are LITs and the
    fallback never triggers (CLAMP stays mono+ in val)."""
    return Expr(opcode=op.OP_CLAMP, children=(_wrap(val), _wrap(lo), _wrap(hi)))


def SP_HEADLINE(skp_idx: int, count_expr) -> Expr:
    """Look up the wynnbuilder "headline" percentage for a skill point
    axis at a given total count. `skp_idx` is one of SKP_STR..SKP_AGI
    from `data.skillpoint_lookup` (0..4). The result is a fraction in
    [0, ~0.808] — wrap in `ADD(LIT(1), SP_HEADLINE(...))` to obtain a
    multiplier of the form (1 + pct).
    """
    if not (0 <= skp_idx <= 4):
        raise ValueError(f"SP_HEADLINE skp_idx must be 0..4, got {skp_idx}")
    return Expr(opcode=op.OP_SP_HEADLINE, arg0=int(skp_idx), children=(_wrap(count_expr),))


def SP_ELEMENT(skp_idx: int, count_expr) -> Expr:
    """Same as SP_HEADLINE but uses the elemental-damage multiplier
    table (str→earth, dex→thunder, int→water, def→fire, agi→air)."""
    if not (0 <= skp_idx <= 4):
        raise ValueError(f"SP_ELEMENT skp_idx must be 0..4, got {skp_idx}")
    return Expr(opcode=op.OP_SP_ELEMENT, arg0=int(skp_idx), children=(_wrap(count_expr),))


def _wrap(x) -> Expr:
    """Auto-promote raw Python numbers to LIT(...) so the user can write
    e.g. `ADD(100, STAT('damPct'))` without manually wrapping."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float)):
        return LIT(x)
    raise TypeError(f"DSL operand must be Expr or numeric, got {type(x).__name__}")


# ============================================================
# Compiled program container
# ============================================================

@dataclass
class Program:
    """A compiled axis program. All arrays are numba-passable."""
    op_codes: np.ndarray   # int32[N]
    op_args:  np.ndarray   # int32[N, 2]
    consts:   np.ndarray   # float64[K]
    # Maximum stack depth required during evaluation. The numba kernel
    # allocates a scratch float64 of this length.
    max_stack: int
    # Whether the program is fully monotone (every opcode has +1/-1
    # tags only). Guides the LB/UB evaluator's fast path vs two-corner
    # fallback.
    is_monotone: bool


# ============================================================
# Compiler
# ============================================================

def compile_program(expr: Expr, query) -> Program:
    """Flatten an Expr AST into a Program ready for the numba evaluator.

    Args:
        expr  : the root Expr.
        query : a `Query` object whose `proj_stats_idx` is consulted to
                resolve STAT() references. STAT('foo') compiles to a
                LOAD_STAT of the projected index for 'foo', and raises
                if the stat is not in the active set.

    For each LOAD_STAT instruction the compiler bakes a "lb_corner_sign"
    into op_args[i, 1] (+1 or -1). For globally-monotone programs the
    LB/UB evaluator uses this to pick which side of [stat_lb, stat_ub] to
    read at this leaf when computing the program's lower/upper bound.
    """
    op_codes: List[int] = []
    op_args:  List[Tuple[int, int]] = []
    consts:   List[float] = []
    const_pool_idx: dict = {}
    cur_depth = 0
    max_depth = 0
    is_monotone = True

    def intern_const(v: float) -> int:
        if v in const_pool_idx:
            return const_pool_idx[v]
        idx = len(consts)
        consts.append(float(v))
        const_pool_idx[v] = idx
        return idx

    def emit(node: Expr, sign: int):
        """Emit `node`'s opcodes in postorder. `sign` is the propagated
        monotonicity sign from the program root (+1 or -1). For non-
        monotone subtrees we mark `is_monotone=False` and continue with
        sign=+1 as a placeholder — the LB/UB evaluator will fall back
        to two-corner evaluation in that case so the per-leaf sign is
        irrelevant."""
        nonlocal cur_depth, max_depth, is_monotone

        # Sign propagation per child, based on the opcode's per-arg
        # monotonicity tags.
        mono = op.MONO_TABLE.get(node.opcode, ())
        if not all(t != 0 for t in mono):
            is_monotone = False
        if mono and len(node.children) > 0:
            if len(mono) == 1:
                child_signs = [sign * mono[0]] * len(node.children)
            else:
                if len(mono) != len(node.children):
                    raise RuntimeError(
                        f"DSL: opcode {node.opcode} expected {len(mono)} children, "
                        f"got {len(node.children)} — MONO_TABLE arity mismatch"
                    )
                child_signs = [sign * m for m in mono]
        else:
            child_signs = []

        for child, child_sign in zip(node.children, child_signs):
            emit(child, child_sign)

        # Resolve operand-side fields and the stored corner sign.
        a0, a1 = -1, -1
        if node.opcode == op.OP_LIT:
            a0 = intern_const(node.lit_value)
        elif node.opcode == op.OP_LOAD_STAT:
            full_idx = STAT_INDEX.get(node.stat_name)
            if full_idx is None:
                raise ValueError(f"DSL STAT('{node.stat_name}'): unknown stat name")
            proj_idx = int(query.proj_stats_idx[full_idx])
            if proj_idx < 0:
                raise ValueError(
                    f"DSL STAT('{node.stat_name}'): not active in this Query "
                    f"— add it to ingredient_filter_stats / hard_constraints."
                )
            a0 = proj_idx
            # Bake the LB-corner side into op_args[i, 1]: +1 means "use
            # stats_lb here for the program's LB", -1 means "use stats_ub
            # for the program's LB". UB-corner is always the opposite.
            a1 = +1 if sign >= 0 else -1
        elif node.opcode == op.OP_LOAD_CTX:
            a0 = int(node.ctx_idx)
        elif node.opcode in (op.OP_ADD, op.OP_MUL, op.OP_MAX, op.OP_MIN):
            a0 = int(node.arg0)
        elif node.opcode in (op.OP_SP_HEADLINE, op.OP_SP_ELEMENT):
            a0 = int(node.arg0)
        # SUB/DIV/NEG/CLAMP need no stored args — operands are stack-only.

        op_codes.append(int(node.opcode))
        op_args.append((a0, a1))

        # Update stack depth.
        popped, pushed = op.STACK_EFFECT[node.opcode]
        if popped < 0:  # arity-n
            popped = int(node.arg0)
        cur_depth -= popped
        if cur_depth < 0:
            raise RuntimeError("DSL compiler: stack underflow (internal bug)")
        cur_depth += pushed
        if cur_depth > max_depth:
            max_depth = cur_depth

    emit(expr, sign=+1)

    if cur_depth != 1:
        raise RuntimeError(
            f"DSL compiler: program left {cur_depth} values on stack at end "
            f"(should be exactly 1). AST root may have invalid arity."
        )

    return Program(
        op_codes=np.asarray(op_codes, dtype=np.int32),
        op_args=np.asarray(op_args if op_args else [(0, 0)], dtype=np.int32).reshape(-1, 2),
        consts=np.asarray(consts if consts else [0.0], dtype=np.float64),
        max_stack=max_depth,
        is_monotone=is_monotone,
    )
