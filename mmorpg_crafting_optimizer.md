# Solving an Ordered MMORPG Crafting Optimization Problem with Slot Efficiencies

## 1. Problem Summary

We want to optimize a crafting system where a player creates one final crafted item by filling 6 ordered ingredient slots.

Each slot must contain exactly one ingredient. Ingredients may be reused, including multiple times in the same craft.

Each ingredient contributes values to a set of stats, for example:

- HP
- ATK
- DEF
- Crit
- Speed
- Resistance
- any other game stat

In the full problem, there may be roughly:

- 200 ingredients
- 6 crafting slots
- 50 stats
- fixed slot efficiencies
- some ingredients that modify the efficiencies of other slots
- user queries with stat minimums, maximums, and score weights

This document explains how to solve the problem mathematically while ignoring composite stats such as `HP_Eff = HP * HP%`.

The final model is an exact mixed-integer linear program, or MILP, as long as all final stats are linear sums of ingredient contributions multiplied by additive slot efficiencies.

---

## 2. Why the Problem Is Not Simple Brute Force

If there are 200 ingredients and 6 ordered slots, then the number of possible ordered crafts is:

```text
200^6 = 64,000,000,000,000
```

That is 64 trillion possible crafts.

Brute force over all ordered recipes is therefore not realistic.

However, the problem has strong structure:

- exactly 6 slots;
- each slot chooses exactly one ingredient;
- ingredients can be reused;
- stats are additive after applying efficiencies;
- efficiency effects are additive;
- the objective is a weighted sum of final stats;
- constraints are stat minimums and maximums.

This structure allows the problem to be solved as an optimization model instead of brute force.

---

## 3. Ingredients and Base Stats

Let:

```text
n = number of ingredients
m = number of stats
P = number of slots = 6
```

For each ingredient `i` and stat `k`, define:

```text
s[k, i] = base contribution of ingredient i to stat k
```

For example:

| Ingredient | HP | DEF | ATK |
|---|---:|---:|---:|
| wood | 20 | 5 | 1 |
| iron | -10 | 20 | 20 |
| diamond | 0 | -50 | 50 |

Stats may be positive, negative, or zero.

Negative stats are allowed. This is why greedy methods are unreliable: an ingredient with excellent ATK may destroy DEF or HP.

---

## 4. Ordered Slots

The 6 slots are ordered.

That means this craft:

```text
slot 1: wood
slot 2: iron
```

may be different from this craft:

```text
slot 1: iron
slot 2: wood
```

because slot efficiencies may differ.

Let slots be indexed by:

```text
p = 1, 2, 3, 4, 5, 6
```

---

## 5. Assignment Variables

Define a binary decision variable:

```text
y[p, i] = 1 if ingredient i is placed in slot p
          0 otherwise
```

Each slot must contain exactly one ingredient:

```text
sum over i of y[p, i] = 1    for every slot p
```

In mathematical notation:

```text
Σ_i y[p,i] = 1    for all p = 1..6
```

Because ingredients can be reused, there is no constraint like:

```text
sum over p of y[p, i] <= 1
```

That constraint would forbid using the same ingredient multiple times, which is not desired.

So the same ingredient can appear in all 6 slots if that is optimal.

---

## 6. Fixed Slot Efficiencies

Suppose each slot has a base fixed efficiency.

Example:

```text
[200%, 100%, 50%, 10%, 500%, 25%]
```

Internally, use multipliers:

```text
[2.0, 1.0, 0.5, 0.1, 5.0, 0.25]
```

Let:

```text
base_eff[p] = fixed base efficiency multiplier of slot p
```

If there were no ingredient-based efficiency modifications, final stat `k` would be:

```text
Stat[k] = Σ_p Σ_i base_eff[p] * s[k,i] * y[p,i]
```

This is linear because `base_eff[p]` and `s[k,i]` are constants.

---

## 7. Ingredient-Based Efficiency Effects

Now add the more difficult rule:

> Some ingredients modify the efficiencies of other slots.

There is no concept of time. Final slot efficiencies are simply the base efficiency plus all efficiency bonuses that apply to that slot.

Let:

```text
b[q, i, p] = efficiency bonus applied to slot p
             when ingredient i is placed in slot q
```

For example:

```text
b[1, ruby, 5] = 2.0
```

means:

```text
if ruby is placed in slot 1, slot 5 gets +200% efficiency
```

Similarly:

```text
b[3, cursed_iron, 5] = -0.5
```

means:

```text
if cursed_iron is placed in slot 3, slot 5 gets -50% efficiency
```

If ingredients cannot affect the slot they are placed in, then:

```text
b[p, i, p] = 0
```

for every slot `p` and ingredient `i`.

Only some ingredients may have nonzero efficiency effects. If roughly 25 out of 200 ingredients affect efficiencies, the model is sparse, which helps performance.

---

## 8. Final Slot Efficiency

Define:

```text
E[p] = final efficiency multiplier of slot p
```

If fixed slot efficiencies exist, then:

```text
E[p] = base_eff[p] + Σ_q Σ_i b[q,i,p] * y[q,i]
```

If every slot starts at 100% and only ingredient bonuses apply, then:

```text
base_eff[p] = 1.0
```

So:

```text
E[p] = 1.0 + Σ_q Σ_i b[q,i,p] * y[q,i]
```

If ingredients only affect other slots, use:

```text
E[p] = base_eff[p] + Σ_{q != p} Σ_i b[q,i,p] * y[q,i]
```

Important: for slot 5, all other 5 slots can contribute to its final efficiency.

For example:

```text
E[5] = base_eff[5]
     + contribution from slot 1
     + contribution from slot 2
     + contribution from slot 3
     + contribution from slot 4
     + contribution from slot 6
```

Mathematically:

```text
E[5] = base_eff[5]
     + Σ_i b[1,i,5] * y[1,i]
     + Σ_i b[2,i,5] * y[2,i]
     + Σ_i b[3,i,5] * y[3,i]
     + Σ_i b[4,i,5] * y[4,i]
     + Σ_i b[6,i,5] * y[6,i]
```

Each source slot chooses exactly one ingredient, so each source slot contributes exactly one selected efficiency effect to slot 5.

---

## 9. The Nonlinear-Looking Part

If ingredient `i` is placed in slot `p`, its stat contribution should be multiplied by final slot efficiency `E[p]`.

So final stat `k` is naturally:

```text
Stat[k] = Σ_p Σ_i s[k,i] * E[p] * y[p,i]
```

The product:

```text
E[p] * y[p,i]
```

is nonlinear because:

- `E[p]` depends on other decision variables;
- `y[p,i]` is a binary decision variable.

However, this product can be linearized exactly.

This is the key step.

---

## 10. Exact Linearization

Introduce a new continuous variable:

```text
u[p, i] = E[p] * y[p, i]
```

Then final stats become linear:

```text
Stat[k] = Σ_p Σ_i s[k,i] * u[p,i]
```

Now we only need to enforce:

```text
u[p,i] = E[p] * y[p,i]
```

exactly using linear constraints.

This is possible because `y[p,i]` is binary.

---

## 11. Bounds on Slot Efficiencies

To linearize `u[p,i] = E[p] * y[p,i]`, we need valid lower and upper bounds for each `E[p]`.

Let:

```text
E_min[p] <= E[p] <= E_max[p]
```

These bounds do not need to be tight for correctness, but tighter bounds improve solver performance.

A safe way to compute them is:

```text
E_min[p] = base_eff[p] + Σ_{q != p} min_i b[q,i,p]
E_max[p] = base_eff[p] + Σ_{q != p} max_i b[q,i,p]
```

Use `q != p` if ingredients cannot affect their own slot.

Use all `q` if self-slot effects are allowed:

```text
E_min[p] = base_eff[p] + Σ_q min_i b[q,i,p]
E_max[p] = base_eff[p] + Σ_q max_i b[q,i,p]
```

The minimum and maximum are taken over all ingredients that can be placed in source slot `q`.

Include ingredients with zero efficiency effects. This matters because zero may be the best or worst available effect.

Example:

If slot 1 can affect slot 5 with possible bonuses:

```text
-50%, 0%, +20%, +100%
```

then:

```text
min_i b[1,i,5] = -0.5
max_i b[1,i,5] = 1.0
```

Do this for every source slot and add the results.

---

## 12. Linearization Constraints

For each slot `p` and ingredient `i`, add these constraints:

```text
u[p,i] <= E_max[p] * y[p,i]

u[p,i] >= E_min[p] * y[p,i]

u[p,i] <= E[p] - E_min[p] * (1 - y[p,i])

u[p,i] >= E[p] - E_max[p] * (1 - y[p,i])
```

These constraints exactly enforce:

```text
u[p,i] = E[p] * y[p,i]
```

when `y[p,i]` is binary.

Why it works:

If `y[p,i] = 0`, the constraints force:

```text
u[p,i] = 0
```

If `y[p,i] = 1`, the constraints force:

```text
u[p,i] = E[p]
```

Therefore the linearized model is not an approximation. It is exactly equivalent to the nonlinear product.

---

## 13. Final Stat Calculation

Once `u[p,i]` is defined, every final stat is linear:

```text
Stat[k] = Σ_p Σ_i s[k,i] * u[p,i]
```

This includes negative stats correctly.

For example, if an ingredient has:

```text
DEF = -50
```

and its slot efficiency is:

```text
E[p] = 2.5
```

then its final DEF contribution is:

```text
-50 * 2.5 = -125
```

The model handles this naturally.

---

## 14. User Query Constraints

A user query may define minimums, maximums, and weights.

Example:

```text
HP  >= 100, weight 0.1
DEF >= 10,  weight 1
ATK >= 0,   weight 10
```

For each stat `k`, let:

```text
L[k] = minimum required value, if any
U[k] = maximum allowed value, if any
w[k] = score weight
```

Minimum constraint:

```text
Stat[k] >= L[k]
```

Maximum constraint:

```text
Stat[k] <= U[k]
```

If a stat has no minimum, omit the lower-bound constraint.

If a stat has no maximum, omit the upper-bound constraint.

---

## 15. Objective Function

The score is a weighted sum of final stats:

```text
Score = Σ_k w[k] * Stat[k]
```

Since:

```text
Stat[k] = Σ_p Σ_i s[k,i] * u[p,i]
```

the objective can be written as:

```text
maximize Σ_k w[k] * Σ_p Σ_i s[k,i] * u[p,i]
```

Equivalently, precompute a base weighted value for each ingredient:

```text
value[i] = Σ_k w[k] * s[k,i]
```

Then:

```text
maximize Σ_p Σ_i value[i] * u[p,i]
```

This works because `u[p,i]` already includes the final slot efficiency.

---

## 16. Full MILP Formulation

### Sets

```text
P = set of slots, usually {1,2,3,4,5,6}
I = set of ingredients
K = set of stats
```

### Constants

```text
s[k,i]       = base contribution of ingredient i to stat k
base_eff[p]  = base efficiency of slot p
b[q,i,p]     = efficiency bonus to slot p if ingredient i is in slot q
L[k]         = minimum allowed value for stat k, optional
U[k]         = maximum allowed value for stat k, optional
w[k]         = score weight for stat k
E_min[p]     = lower bound for final efficiency of slot p
E_max[p]     = upper bound for final efficiency of slot p
```

### Variables

```text
y[p,i] ∈ {0,1}
E[p]   continuous
u[p,i] continuous
Stat[k] continuous
```

### Slot assignment

```text
Σ_i y[p,i] = 1                    for every slot p
```

### Final slot efficiencies

If ingredients cannot affect their own slot:

```text
E[p] = base_eff[p] + Σ_{q != p} Σ_i b[q,i,p] * y[q,i]
```

If self-slot effects are allowed:

```text
E[p] = base_eff[p] + Σ_q Σ_i b[q,i,p] * y[q,i]
```

### Linearization

For every `p,i`:

```text
u[p,i] <= E_max[p] * y[p,i]

u[p,i] >= E_min[p] * y[p,i]

u[p,i] <= E[p] - E_min[p] * (1 - y[p,i])

u[p,i] >= E[p] - E_max[p] * (1 - y[p,i])
```

### Final stats

```text
Stat[k] = Σ_p Σ_i s[k,i] * u[p,i]    for every stat k
```

### Query constraints

For every stat with a minimum:

```text
Stat[k] >= L[k]
```

For every stat with a maximum:

```text
Stat[k] <= U[k]
```

### Objective

```text
maximize Σ_k w[k] * Stat[k]
```

This is a mixed-integer linear program.

---

## 17. Why the Model Is Exact

The only nonlinear-looking rule is:

```text
ingredient contribution = base stat * final slot efficiency
```

This creates:

```text
E[p] * y[p,i]
```

The linearization using `u[p,i]` is exact because `y[p,i]` is binary.

The model is not approximating the game rules.

If the solver proves optimality, then the returned craft is globally optimal for the stated query.

Exactness depends on these assumptions:

1. Every slot is filled by exactly one ingredient.
2. Ingredients can be reused.
3. Efficiency bonuses are additive.
4. Final stats are sums of slot-adjusted ingredient stats.
5. The score is a linear weighted sum of final stats.
6. Composite stats such as `HP * HP%` are ignored.
7. The solver is allowed to run until it proves optimality.

---

## 18. Expected Model Size

For approximately:

```text
6 slots
200 ingredients
50 stats
```

the model has roughly:

| Component | Count |
|---|---:|
| Binary assignment variables `y[p,i]` | 6 × 200 = 1200 |
| Efficiency variables `E[p]` | 6 |
| Auxiliary variables `u[p,i]` | 6 × 200 = 1200 |
| Final stat variables `Stat[k]` | 50 |
| Slot assignment constraints | 6 |
| Efficiency equations | 6 |
| Linearization constraints | 4 × 1200 = 4800 |
| Stat equations | 50 |
| Min/max query constraints | up to 100 |

This is a reasonable MILP size.

The problem is harder than a purely linear no-interaction version, but the small number of slots keeps it practical.

If only around 25 ingredients affect efficiencies, the efficiency equations are sparse, which helps significantly.

---

## 19. Scaling Percentages Safely

Avoid floating-point ambiguity if possible.

Instead of storing:

```text
100% = 1.0
200% = 2.0
25%  = 0.25
```

use integer scaling.

For example, basis points:

```text
100% = 10000
200% = 20000
25%  = 2500
10%  = 1000
```

Then efficiency bonuses are integers too:

```text
+50% = +5000
-25% = -2500
```

If you use scaled efficiencies, remember that final stats are also scaled.

For example:

```text
raw_final_stat = Σ s[k,i] * u[p,i]
actual_final_stat = raw_final_stat / 10000
```

You can either:

- divide at the end; or
- scale query thresholds and weights consistently.

Integer scaling usually makes implementation cleaner and avoids precision surprises.

---

## 20. Handling Infeasible Queries

A query may have no valid craft.

For example:

```text
HP >= 10000
DEF >= 10000
ATK >= 10000
```

may be impossible with only 6 ingredients.

The solver should be allowed to return:

```text
infeasible
```

This is not an error. It means no craft satisfies all minimum and maximum constraints.

A useful application should report this clearly to the user.

Possible user-facing messages:

```text
No craft satisfies all requested constraints.
```

or:

```text
The requested minimums/maximums are too strict for 6 ingredients.
```

You may also implement a relaxation mode that finds the closest craft, but that is a separate optimization problem.

---

## 21. Optional: Soft Constraints

Sometimes a user may prefer a result even if the exact query is impossible.

You can introduce slack variables.

For a minimum constraint:

```text
Stat[k] + shortfall[k] >= L[k]
shortfall[k] >= 0
```

Then penalize shortfall in the objective:

```text
maximize score - big_penalty * Σ shortfall[k]
```

For a maximum constraint:

```text
Stat[k] - excess[k] <= U[k]
excess[k] >= 0
```

Then penalize excess:

```text
maximize score - big_penalty * Σ excess[k]
```

This gives a best-effort craft when the strict query is infeasible.

However, for exact query solving, use hard constraints instead.

---

## 22. Practical Solver Choices

The mathematical model is a MILP.

Any capable MILP or CP-SAT solver can be used.

Possible solver families:

- MILP solvers
- CP-SAT solvers
- branch-and-bound solvers
- branch-and-cut solvers

The important requirement is that the solver supports:

- binary variables;
- continuous or integer variables;
- linear equality and inequality constraints;
- linear objective;
- proven optimality status.

For implementation, you should check whether the solver returns one of these statuses:

```text
optimal
feasible but not proven optimal
infeasible
time limit reached
numerical issue
```

Only `optimal` means the craft is mathematically proven to be the best.

---

## 23. Suggested Implementation Pipeline

### Step 1: Load data

Load:

```text
ingredients
base stats
fixed slot efficiencies
ingredient efficiency effects
```

### Step 2: Convert percentages

Convert all percentages to multipliers or scaled integers.

Example:

```text
200% -> 2.0
50%  -> 0.5
```

or:

```text
200% -> 20000
50%  -> 5000
```

### Step 3: Build stat matrix

Create:

```text
s[k,i]
```

for every stat and ingredient.

### Step 4: Build efficiency effect tensor

Create:

```text
b[q,i,p]
```

for every source slot, ingredient, and target slot.

Most values will often be zero.

Store sparsely if possible.

### Step 5: Compute efficiency bounds

For each target slot `p`:

```text
E_min[p] = base_eff[p] + Σ_{q != p} min_i b[q,i,p]
E_max[p] = base_eff[p] + Σ_{q != p} max_i b[q,i,p]
```

### Step 6: Create variables

Create:

```text
y[p,i]
E[p]
u[p,i]
Stat[k]
```

### Step 7: Add constraints

Add:

```text
slot assignment constraints
efficiency equations
linearization constraints
stat equations
query min/max constraints
```

### Step 8: Set objective

Use:

```text
maximize Σ_k w[k] * Stat[k]
```

### Step 9: Solve

Run the solver.

### Step 10: Decode recipe

For each slot `p`, find the ingredient `i` such that:

```text
y[p,i] = 1
```

Return the ordered craft:

```text
slot 1 -> ingredient
slot 2 -> ingredient
...
slot 6 -> ingredient
```

Also return:

```text
final slot efficiencies
final stats
score
solver status
```

---

## 24. Pseudocode

```text
input:
  ingredients I
  stats K
  slots P = {1,2,3,4,5,6}
  base stat matrix s[k,i]
  base slot efficiencies base_eff[p]
  efficiency bonuses b[q,i,p]
  query minimums L[k]
  query maximums U[k]
  weights w[k]

compute E_min[p], E_max[p] for every slot p

create MILP model

for each p in P:
  for each i in I:
    create binary y[p,i]
    create continuous u[p,i]

for each p in P:
  create continuous E[p]

for each k in K:
  create continuous Stat[k]

for each p in P:
  add constraint sum_i y[p,i] = 1

for each p in P:
  add constraint:
    E[p] = base_eff[p] + sum_{q != p} sum_i b[q,i,p] * y[q,i]

for each p in P:
  for each i in I:
    add u[p,i] <= E_max[p] * y[p,i]
    add u[p,i] >= E_min[p] * y[p,i]
    add u[p,i] <= E[p] - E_min[p] * (1 - y[p,i])
    add u[p,i] >= E[p] - E_max[p] * (1 - y[p,i])

for each k in K:
  add constraint:
    Stat[k] = sum_p sum_i s[k,i] * u[p,i]

for each stat k with minimum L[k]:
  add Stat[k] >= L[k]

for each stat k with maximum U[k]:
  add Stat[k] <= U[k]

objective:
  maximize sum_k w[k] * Stat[k]

solve model

if status is optimal:
  return selected ingredient in each slot, final stats, efficiencies, score
else if status is infeasible:
  report no valid craft
else:
  report best known craft if available, but not proven optimal
```

---

## 25. Worked Mini Example

Suppose there are 3 ingredients and 3 stats.

| Ingredient | HP | DEF | ATK |
|---|---:|---:|---:|
| wood | 20 | 5 | 1 |
| iron | -10 | 20 | 20 |
| diamond | 0 | -50 | 50 |

Suppose there are 6 slots with base efficiencies:

```text
[200%, 100%, 50%, 10%, 500%, 25%]
```

As multipliers:

```text
[2.0, 1.0, 0.5, 0.1, 5.0, 0.25]
```

Suppose iron in slot 1 gives slot 5 `+100%` efficiency:

```text
b[1, iron, 5] = 1.0
```

Suppose diamond in slot 3 gives slot 5 `-50%` efficiency:

```text
b[3, diamond, 5] = -0.5
```

Then if:

```text
slot 1 = iron
slot 3 = diamond
```

slot 5 efficiency is:

```text
E[5] = base_eff[5] + 1.0 - 0.5
     = 5.0 + 1.0 - 0.5
     = 5.5
```

So slot 5 has:

```text
550% efficiency
```

If wood is placed in slot 5, its final contribution is:

```text
HP  = 20 * 5.5 = 110
DEF = 5  * 5.5 = 27.5
ATK = 1  * 5.5 = 5.5
```

The optimization model decides both:

1. which ingredients create useful slot efficiencies;
2. which ingredients should be placed into the boosted slots.

---

## 26. Common Pitfalls

### Pitfall 1: Treating order as irrelevant

Order matters because slot efficiencies differ and can be modified.

Use assignment variables `y[p,i]`, not just ingredient counts `x[i]`.

### Pitfall 2: Preventing ingredient reuse by mistake

Do not add:

```text
Σ_p y[p,i] <= 1
```

unless the game actually forbids reuse.

### Pitfall 3: Forgetting that all other slots can affect a slot

For a target slot `p`, every other slot can contribute to its final efficiency.

Use:

```text
E[p] = base_eff[p] + Σ_{q != p} Σ_i b[q,i,p] * y[q,i]
```

not only one source slot.

### Pitfall 4: Using invalid efficiency bounds

The linearization is exact only if `E_min[p]` and `E_max[p]` are valid.

They may be loose, but they must contain all possible values of `E[p]`.

### Pitfall 5: Stopping the solver early and calling it optimal

If the solver returns a feasible solution without proof of optimality, the craft may be good but is not mathematically guaranteed best.

### Pitfall 6: Mixing percentages and multipliers

Be consistent.

Either use:

```text
100% = 1.0
```

or:

```text
100% = 10000
```

Do not mix representations.

---

## 27. What This Model Does Not Cover

This document intentionally ignores composite stats.

Examples of composite stats:

```text
Effective HP = HP * HP%
Damage = ATK * CritMultiplier
Survivability = HP * DEF / incoming_damage
```

Those introduce nonlinear relationships between final stats.

The model in this document assumes the final score is a linear weighted sum:

```text
Score = Σ_k w[k] * Stat[k]
```

and that each `Stat[k]` is computed from additive ingredient contributions multiplied by additive slot efficiencies.

Under those assumptions, the MILP formulation is exact.

---

## 28. Final Takeaway

The ordered crafting problem with reusable ingredients and additive slot efficiency effects can be solved exactly as a mixed-integer linear program.

The key modeling choices are:

```text
y[p,i] = whether ingredient i is placed in slot p
E[p]   = final efficiency of slot p
u[p,i] = E[p] * y[p,i]
```

The product `E[p] * y[p,i]` is linearized exactly because `y[p,i]` is binary.

The final model is:

```text
choose exactly one ingredient per slot
compute final slot efficiencies from selected ingredients
compute final stats from selected ingredients and efficiencies
respect user min/max stat constraints
maximize weighted score
```

For roughly 200 ingredients, 6 slots, 50 stats, and sparse efficiency effects, this is a realistic exact optimization model.

