"""
DSL opcode definitions for axis-evaluation programs.

Each axis used by the Pareto-precalc search is compiled to a flat
sequence of these opcodes. The numba evaluator (`dsl.eval`) interprets
the program in a tight loop; the LB/UB evaluator uses the per-opcode
monotonicity declarations below to compute branch-and-bound corners
during DFS.

Storage layout
--------------
A program is three parallel arrays plus a constant pool:
    op_codes : int32[N]      — opcode ID, one of the OP_* constants below
    op_args  : int32[N, 2]   — up to 2 int args per opcode (meanings below)
    consts   : float64[K]    — values referenced by OP_LIT
The evaluator keeps a small float64 scratch stack; opcodes pop their
inputs from the top of the stack and push their result back. Programs
are emitted in postfix (RPN) order by the compiler.

Operand conventions
-------------------
- arity-N reducers (ADD/MUL/MAX/MIN) store N in op_args[i, 0]; the N
  topmost stack values are popped and the result is pushed.
- LIT stores the const-pool index in op_args[i, 0].
- LOAD_STAT stores the projected stat index (the same index space the
  rest of the search uses — `Query.proj_stats_idx`).
- LOAD_CTX stores the BUILD_CTX_* slot index.
- SP_HEADLINE / SP_ELEMENT store the skp axis index (0..4 for
  str/dex/int/def/agi) in op_args[i, 0]. They pop one float (the raw
  skill-point count to look up); the lookup clamps to [0, SKP_MAX] and
  pushes the percentage as a fraction in [0, ~0.808].

Monotonicity
------------
For each opcode we declare a per-arg monotonicity tag in MONO_TABLE.
The DFS bounds use these tags to decide which corner of the
[stat_lb, stat_ub] hypercube to evaluate. Tags:
    +1  : monotone non-decreasing in this argument (use LB-corner for LB,
          UB-corner for UB)
    -1  : monotone non-increasing (corners swap)
     0  : non-monotone — the LB/UB evaluator falls back to evaluating at
          both corners and taking min/max. This is conservative but valid.

For binary/n-ary opcodes whose monotonicity in one argument depends on
the SIGN of another (e.g. MUL when an arg is negative), the table
declares the safe (mono+) case; programs that may produce negative
intermediate values should structure expressions to keep them in the
non-negative regime, or use NON_MONOTONE_PROGRAM as an escape hatch on
the program as a whole.
"""

# ============================================================
# Opcode IDs (kept stable — index into MONO_TABLE)
# ============================================================
# Stack effects below: "n→1" means pops n and pushes 1.

OP_LIT       = 0   # 0→1   pushes consts[arg0]
OP_LOAD_STAT = 1   # 0→1   pushes stats_proj[arg0]
OP_LOAD_CTX  = 2   # 0→1   pushes ctx[arg0]
OP_ADD       = 3   # n→1   arg0 = arity n, sums top n stack values
OP_MUL       = 4   # n→1   arg0 = arity n, multiplies top n stack values
OP_SUB       = 5   # 2→1   pops (b, a), pushes a - b (a was pushed first)
OP_DIV       = 6   # 2→1   pops (b, a), pushes a / b (a was pushed first)
OP_NEG       = 7   # 1→1   negates top
OP_MAX       = 8   # n→1   arg0 = arity n
OP_MIN       = 9   # n→1   arg0 = arity n
OP_CLAMP     = 10  # 3→1   pops (hi, lo, val), pushes min(hi, max(lo, val))
OP_SP_HEADLINE = 11  # 1→1 arg0 = skp_idx (0..4); pops skp_count, pushes
                     #     SKP_HEADLINE_PCT[skp_idx, clamp(skp_count, 0..150)]
OP_SP_ELEMENT  = 12  # 1→1 arg0 = skp_idx; pops skp_count, pushes
                     #     SKP_ELEMENT_PCT[skp_idx, clamp(skp_count, 0..150)]

OPCODE_COUNT = 13


# ============================================================
# Per-argument monotonicity declarations
# ============================================================
# MONO_TABLE[op] = tuple of monotonicity tags, one per argument the
# opcode pops from the stack (in the order they were pushed). LIT,
# LOAD_STAT, LOAD_CTX have no popped args (they push from external
# state). For arity-n opcodes the same tag applies to every popped arg.
#
# +1 = mono non-decreasing, -1 = mono non-increasing, 0 = non-monotone.

MONO_TABLE = {
    OP_LIT:         (),
    OP_LOAD_STAT:   (),
    OP_LOAD_CTX:    (),
    # Tags are indexed by AST CHILD POSITION (matching the Python builder's
    # argument order), not by stack pop order.
    OP_ADD:         (+1,),  # uniform across arity
    OP_MUL:         (+1,),  # ASSUMES non-negative regime; see module docstring
    OP_SUB:         (+1, -1),  # SUB(a, b) = a - b: mono+ in a, mono- in b
    OP_DIV:         (+1, -1),  # DIV(a, b) = a / b (assumes b > 0)
    OP_NEG:         (-1,),
    OP_MAX:         (+1,),
    OP_MIN:         (+1,),
    OP_CLAMP:       (+1, +1, +1),  # CLAMP(val, lo, hi): mono+ in all three
    OP_SP_HEADLINE: (+1,),  # SKP_HEADLINE_PCT is non-decreasing in skp_count
    OP_SP_ELEMENT:  (+1,),
}


# ============================================================
# Stack effect (popped, pushed) per opcode
# ============================================================
# For arity-n opcodes (ADD/MUL/MAX/MIN) the popped count is dynamic and
# read from op_args[i, 0]; STACK_EFFECT marks them with -1 to signal
# that.

STACK_EFFECT = {
    OP_LIT:         (0, 1),
    OP_LOAD_STAT:   (0, 1),
    OP_LOAD_CTX:    (0, 1),
    OP_ADD:         (-1, 1),
    OP_MUL:         (-1, 1),
    OP_SUB:         (2, 1),
    OP_DIV:         (2, 1),
    OP_NEG:         (1, 1),
    OP_MAX:         (-1, 1),
    OP_MIN:         (-1, 1),
    OP_CLAMP:       (3, 1),
    OP_SP_HEADLINE: (1, 1),
    OP_SP_ELEMENT:  (1, 1),
}


# ============================================================
# Helpers for the LB/UB evaluator
# ============================================================
# A program is "globally monotone" if every opcode in it has only +1/-1
# monotonicity tags. Such programs admit the cheap "single-corner per
# bound" evaluation: LB(prog) = prog(stats_lb_corner), where the corner
# is built per-stat by picking lb or ub based on the stat's net sign of
# influence (computed during compilation). Programs containing a
# non-monotone opcode (CLAMP between two non-constant args, etc.) fall
# back to evaluating at both extreme corners and taking min/max.

def is_opcode_monotone(op: int) -> bool:
    """True iff this opcode's monotonicity is fully declared (no 0 tags)."""
    if op not in MONO_TABLE:
        return False
    return all(t != 0 for t in MONO_TABLE[op])
