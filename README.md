# DisjunctiveAlgorithms.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://infiniteopt.github.io/DisjunctiveAlgorithms.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://infiniteopt.github.io/DisjunctiveAlgorithms.jl/dev/)
[![Build Status](https://github.com/infiniteopt/DisjunctiveAlgorithms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/infiniteopt/DisjunctiveAlgorithms.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/infiniteopt/DisjunctiveAlgorithms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/infiniteopt/DisjunctiveAlgorithms.jl)

An optimizer suite for generalized disjunctive programming (GDP).

DisjunctiveAlgorithms.jl is an MOI-layer solver for models that contain
disjunctions encoded as vector constraints in
`DisjunctiveProgramming.DisjunctionSet`. Disjunction-aware algorithms
(currently logic-based outer approximation) solve the model by
dispatching subproblems to user-provided MIP and NLP solvers.

The design follows
[MultiObjectiveAlgorithms.jl](https://github.com/jump-dev/MultiObjectiveAlgorithms.jl):
one `Optimizer` that wraps inner solvers, with the algorithm and its
options selected through optimizer attributes.

## Installation

DisjunctiveAlgorithms needs the `DisjunctionSet` set and the `Direct()`
reformulation, which live on DisjunctiveProgramming's `gdp_optimizer`
branch and are not in a registered release yet. Until they ship,
install the branch first:

```julia
import Pkg
Pkg.add(url = "https://github.com/dnguyen227/DisjunctiveProgramming.jl",
    rev = "gdp_optimizer")
Pkg.add(url = "https://github.com/infiniteopt/DisjunctiveAlgorithms.jl")
```

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
