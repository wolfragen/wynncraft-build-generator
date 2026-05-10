"""
DSL for axis-evaluation programs in the Pareto-precalc search mode.

Public API:
    LIT, STAT, CTX  — leaf builders
    ADD, MUL, SUB, DIV, NEG, MAXOP, MINOP, CLAMP — arithmetic
    SP_HEADLINE, SP_ELEMENT — wynnbuilder skill-point lookups
    compile_program(expr, query) -> Program
    Program (dataclass)
    eval_program, eval_program_bounds — numba evaluators
    SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64 — constants for numba calls
"""
from .builder import (
    Expr, Program,
    LIT, STAT, CTX,
    ADD, MUL, SUB, DIV, NEG, MAXOP, MINOP, CLAMP,
    SP_HEADLINE, SP_ELEMENT,
    compile_program,
)
from .evaluator import (
    eval_program, eval_program_bounds,
    SKP_HEADLINE_PCT_F64, SKP_ELEMENT_PCT_F64,
)
from . import opcodes
