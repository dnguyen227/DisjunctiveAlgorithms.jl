```@meta
CurrentModule = DisjunctiveAlgorithms
```

# DisjunctiveAlgorithms.jl

An optimizer suite for generalized disjunctive programming (GDP).

DisjunctiveAlgorithms.jl is an MOI-layer solver for models that contain
disjunctions encoded as vector constraints in
`DisjunctiveProgramming.DisjunctionSet`. Disjunction-aware algorithms
(currently logic-based outer approximation) solve the model by
dispatching subproblems to user-provided MIP and NLP solvers.

## Installation

DisjunctiveAlgorithms is registered in the General registry. It
requires DisjunctiveProgramming 0.7 or later, which ships the
`DisjunctionSet` set and the `Direct()` reformulation; the package
manager resolves that version automatically.

```julia
import Pkg
Pkg.add("DisjunctiveAlgorithms")
```

Supported Julia versions: 1.10 and later.

## Usage with DisjunctiveProgramming.jl

```julia
using DisjunctiveProgramming, DisjunctiveAlgorithms, HiGHS, Ipopt
import DisjunctiveAlgorithms as DA

model = GDPModel(() -> DA.Optimizer(Ipopt.Optimizer, HiGHS.Optimizer))
@variable(model, 0 <= x <= 10)
@variable(model, Y[1:2], Logical)
@constraint(model, x <= 3, Disjunct(Y[1]))
@constraint(model, x^2 == 64, Disjunct(Y[2]))
@disjunction(model, Y)
@objective(model, Max, x)
optimize!(model, gdp_method = Direct())
```

`Direct()` lowers each disjunction to a single
`DisjunctionSet` constraint that this package consumes directly; no
Big-M or Hull reformulation is performed on the modeling side.

## Options

The algorithm is selected with the [`Algorithm`](@ref) attribute
(default [`LOA`](@ref)`()`) and configured through the attributes it
documents, set with `set_attribute`:

```julia
set_attribute(model, DA.NumIterationLimit(), 20)
set_attribute(model, DA.ConvergenceTolerance(), 1e-4)
```

The [`LOA`](@ref) docstring lists every supported attribute. After a
solve, [`MasterSolveCount`](@ref), [`NLPSolveCount`](@ref),
[`MasterSolveTime`](@ref) and [`NLPSolveTime`](@ref) report the work
done alongside the standard MOI result attributes. The
[API Reference](@ref) lists every exported function and attribute.
