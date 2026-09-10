################################################################################
#                          ALGORITHMS AND ATTRIBUTES
################################################################################
"""
    AbstractAlgorithm

A super-type for the solution algorithms. Select one with the
[`Algorithm`](@ref) attribute.
"""
abstract type AbstractAlgorithm end

"""
    Algorithm() <: MOI.AbstractOptimizerAttribute

The algorithm the optimizer runs. Defaults to [`LOA`](@ref).

**Example**
```julia
julia> set_attribute(model, DisjunctiveAlgorithms.Algorithm(),
           DisjunctiveAlgorithms.LOA())
```
"""
struct Algorithm <: MOI.AbstractOptimizerAttribute end

"""
    AbstractAlgorithmAttribute <: MOI.AbstractOptimizerAttribute

A super-type for algorithm-specific optimizer attributes. Each
algorithm declares `MOI.supports` for the attributes it consumes and
documents them in its docstring.
"""
abstract type AbstractAlgorithmAttribute <: MOI.AbstractOptimizerAttribute end

_default(::AbstractAlgorithm, attr::AbstractAlgorithmAttribute) =
    _default(attr)

"""
    NumIterationLimit() <: AbstractAlgorithmAttribute -> Int

Master/NLP iterations after the set-covering seed. Defaults to `10`.
"""
struct NumIterationLimit <: AbstractAlgorithmAttribute end
_default(::NumIterationLimit) = 10

"""
    SetCoverIterationLimit() <: AbstractAlgorithmAttribute -> Int

Set-covering initialization iterations. Defaults to `8`.
"""
struct SetCoverIterationLimit <: AbstractAlgorithmAttribute end
_default(::SetCoverIterationLimit) = 8

"""
    M_Value() <: AbstractAlgorithmAttribute -> Float64

Big-M gating the disjunct OA cuts in the master. Defaults to `1e9`.
"""
struct M_Value <: AbstractAlgorithmAttribute end
_default(::M_Value) = 1e9

"""
    MasterReformulation() <: AbstractAlgorithmAttribute -> String

How linear disjunct rows enter the master: `"indicator"` (constraint
gated by the binary) or `"bigm"` (rows relaxed by `M_Value * (1 - z)`,
tighter for solvers that cannot strengthen indicators, e.g. with
presolve disabled). Defaults to `"indicator"`.
"""
struct MasterReformulation <: AbstractAlgorithmAttribute end
_default(::MasterReformulation) = "indicator"

"""
    MaxSlack() <: AbstractAlgorithmAttribute -> Float64

Upper bound of each OA cut slack. Defaults to `1e3`.
"""
struct MaxSlack <: AbstractAlgorithmAttribute end
_default(::MaxSlack) = 1e3

"""
    OASlack() <: AbstractAlgorithmAttribute -> Float64

Objective penalty per unit of cut slack. Defaults to `1e3`.
"""
struct OASlack <: AbstractAlgorithmAttribute end
_default(::OASlack) = 1e3

"""
    UseNLPF() <: AbstractAlgorithmAttribute -> Bool

Solve a slacked feasibility NLP when the primary NLP is infeasible,
so its point still seeds OA cuts. Defaults to `true`.
"""
struct UseNLPF <: AbstractAlgorithmAttribute end
_default(::UseNLPF) = true

"""
    ConvergenceTolerance() <: AbstractAlgorithmAttribute -> Float64

Relative incumbent/bound gap tolerance. Defaults to `1e-6`.
"""
struct ConvergenceTolerance <: AbstractAlgorithmAttribute end
_default(::ConvergenceTolerance) = 1e-6

"""
    SlackTolerance() <: AbstractAlgorithmAttribute -> Float64

Total cut slack tolerance for convergence. Defaults to `1e-4`.
"""
struct SlackTolerance <: AbstractAlgorithmAttribute end
_default(::SlackTolerance) = 1e-4

"""
    IterationTimeLimit() <: AbstractAlgorithmAttribute -> Float64

Seconds allotted to the solve loop. Defaults to `Inf`. The overall
budget is the standard `MOI.TimeLimitSec` attribute.
"""
struct IterationTimeLimit <: AbstractAlgorithmAttribute end
_default(::IterationTimeLimit) = Inf

"""
    MultiGenerationSize() <: AbstractAlgorithmAttribute -> Int

Indicator combinations evaluated per master solve in the main loop
(multi-generation cuts). Defaults to `1`, plain LOA. Extra
combinations come from the master's solution pool when the
`mip_solver` exposes one through `MOI.ResultCount` (Gurobi with
`PoolSolutions`), and otherwise from re-solving the master behind a
no-good cut per combination already taken. All of them go through
`_solve_nlps`, one stacked NLP under [`BatchedSubproblems`](@ref), and
every result adds its cuts before the next master solve.
"""
struct MultiGenerationSize <: AbstractAlgorithmAttribute end
_default(::MultiGenerationSize) = 1

"""
    SubproblemMethod() <: AbstractAlgorithmAttribute -> Any

How the NLP subproblems are built and solved. The default `nothing`
builds one MOI model from the `nlp_solver` and rewrites its binary
fixes and disjunct rows per combination. A method object routes the
subproblems elsewhere (e.g. a structure-preserving transcription of
the original model): it must extend `_build_subproblem` and
`_solve_nlp` on its type, and may extend `_check_nlp_support`.
`_solve_nlp` must return `(; combination, point, objective, feasible,
status)` with `point` indexed by this optimizer's variable indices.
"""
struct SubproblemMethod <: AbstractAlgorithmAttribute end
_default(::SubproblemMethod) = nothing

################################################################################
#                                OPTIMIZER
################################################################################
const _Cache = MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}

"""
    Optimizer(nlp_solver, mip_solver = nlp_solver)

Solver for models containing
`DisjunctiveProgramming.DisjunctionSet` constraints. `nlp_solver`
and `mip_solver` are optimizer factories as accepted by
`MOI.instantiate`. The algorithm (default [`LOA`](@ref)) is selected
with the [`Algorithm`](@ref) attribute and configured through the
attributes it documents. A new optimizer starts with a 3600 second
`MOI.TimeLimitSec` safety limit; set it to `nothing` for no limit.

**Example**
```julia
julia> model = GDPModel(() -> DisjunctiveAlgorithms.Optimizer(
           Ipopt.Optimizer, HiGHS.Optimizer));

julia> optimize!(model, gdp_method = Direct())
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    nlp_solver::Any
    mip_solver::Any
    cache::_Cache
    algorithm::Union{Nothing, AbstractAlgorithm}
    time_limit_sec::Union{Nothing, Float64}
    silent::Bool
    # results (filled by MOI.optimize!)
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    incumbent::Dict{MOI.VariableIndex, Float64}
    objective_value::Float64
    objective_bound::Union{Nothing, Float64}
    relative_gap::Float64
    raw_status::String
    solve_time::Float64
    num_master_solves::Int
    num_nlp_solves::Int
end

function Optimizer(nlp_solver, mip_solver = nlp_solver)
    return Optimizer(nlp_solver, mip_solver,
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        nothing, 3600.0, false, MOI.OPTIMIZE_NOT_CALLED, MOI.NO_SOLUTION,
        Dict{MOI.VariableIndex, Float64}(), NaN, nothing, NaN, "", NaN, 0, 0)
end

_algorithm(model::Optimizer) =
    something(model.algorithm, _default(Algorithm()))

MOI.get(::Optimizer, ::MOI.SolverName) = "DisjunctiveAlgorithms"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(@__MODULE__))

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.cache)

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.cache)
    _reset_results(model)
    return
end

# The non-cache part of MOI.empty!, also run at the top of every solve
# so a re-optimize cannot leak the previous solve's bound, gap, or
# point.
function _reset_results(model::Optimizer)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.primal_status = MOI.NO_SOLUTION
    empty!(model.incumbent)
    model.objective_value = NaN
    model.objective_bound = nothing
    model.relative_gap = NaN
    model.raw_status = ""
    model.solve_time = NaN
    model.num_master_solves = 0
    model.num_nlp_solves = 0
    return
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.optimize!(model::Optimizer) = _optimize!(_algorithm(model), model)

################################################################################
#                        MODEL-BUILDING FORWARDING
################################################################################
# Everything model-building lands in the cache; the solve partitions it
# in MOI.optimize!, so incremental additions need no invalidation.
const _Index = Union{MOI.VariableIndex, MOI.ConstraintIndex}

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.cache)

MOI.add_variables(model::Optimizer, n::Int) = MOI.add_variables(model.cache, n)

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.AbstractScalarSet
    )
    return MOI.add_constrained_variable(model.cache, set)
end

function MOI.add_constrained_variables(
    model::Optimizer,
    set::MOI.AbstractVectorSet
    )
    return MOI.add_constrained_variables(model.cache, set)
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet
    )
    return MOI.add_constraint(model.cache, func, set)
end

# Honest constraint support (the cache would claim everything, which
# disables the bridges that rewrite e.g. vector cones into supported
# scalar rows): only what `_build_problem` actually partitions.
const _ScalarFunction = Union{MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}, MOI.ScalarNonlinearFunction}
const _VectorFunction = Union{MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    MOI.VectorQuadraticFunction{Float64}, MOI.VectorNonlinearFunction}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.AbstractFunction},
    ::Type{<:MOI.AbstractSet}
    )
    return false
end

# Non-indicator discrete variables pass through to the subproblems
# with their integrality intact, so the nlp_solver must handle them.
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{SupportedInnerSet, MOI.ZeroOne, MOI.Integer}}
    )
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_ScalarFunction},
    ::Type{<:SupportedInnerSet}
    )
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_VectorFunction},
    ::Type{DisjunctionSet}
    )
    return true
end

# A `DisjunctionSet` never constrains variables on creation: without
# this, `copy_to` turns a `VectorOfVariables` disjunction (which may
# repeat a variable across rows) into `add_constrained_variables`.
function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{DisjunctionSet}
    )
    return false
end

MOI.is_valid(model::Optimizer, index::_Index) =
    MOI.is_valid(model.cache, index)

MOI.delete(model::Optimizer, index::_Index) = MOI.delete(model.cache, index)

function MOI.get(model::Optimizer, T::Type{<:_Index}, name::String)
    return MOI.get(model.cache, T, name)
end

# Non-result attributes forward to the cache; the result attributes
# below override these generic methods, and any solve-set attribute
# without an override (duals, ConstraintPrimal) is refused rather than
# forwarded to the cache, which never holds results.
# Forward to the wrapped inner model (honest), not the UniversalFallback,
# which claims support for every attribute and so never rejects an
# unsupported one on copy_to.
MOI.supports(model::Optimizer, attr::MOI.AbstractModelAttribute) =
    MOI.supports(model.cache.model, attr)

# Objective functions `_build_problem` can consume; the generic
# forwarding above would claim vector objectives the solve rejects.
const _ObjectiveFunction = Union{MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}, MOI.ScalarNonlinearFunction}

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return F <: _ObjectiveFunction
end

function MOI.set(model::Optimizer, attr::MOI.AbstractModelAttribute, value)
    MOI.supports(model, attr) || throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, value)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex}
    )
    return MOI.supports(model.cache.model, attr, MOI.VariableIndex)
end

# The LOA loop consumes warm starts, so accept them even though the
# inner model does not store them (the fallback does).
function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex}
    )
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value
    )
    MOI.supports(model, attr, MOI.VariableIndex) ||
        throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, vi, value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex
    )
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr, vi)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    C::Type{<:MOI.ConstraintIndex}
    )
    return MOI.supports(model.cache.model, attr, C)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value
    )
    MOI.supports(model, attr, typeof(ci)) ||
        throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, ci, value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex
    )
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr, ci)
end

################################################################################
#                          OPTIMIZER ATTRIBUTES
################################################################################
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(
    model::Optimizer,
    ::MOI.TimeLimitSec,
    value::Union{Nothing, Real}
    )
    model.time_limit_sec = value === nothing ? nothing : Float64(value)
    return
end

MOI.get(model::Optimizer, ::MOI.TimeLimitSec) = model.time_limit_sec

MOI.supports(::Optimizer, ::Algorithm) = true

MOI.get(model::Optimizer, ::Algorithm) = model.algorithm

function MOI.set(model::Optimizer, ::Algorithm, algorithm::AbstractAlgorithm)
    model.algorithm = algorithm
    return
end

function MOI.supports(model::Optimizer, attr::AbstractAlgorithmAttribute)
    return MOI.supports(_algorithm(model), attr)
end

function MOI.set(model::Optimizer, attr::AbstractAlgorithmAttribute, value)
    model.algorithm === nothing && (model.algorithm = _default(Algorithm()))
    MOI.set(model.algorithm, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::AbstractAlgorithmAttribute)
    return MOI.get(_algorithm(model), attr)
end

################################################################################
#                           RESULT ATTRIBUTES
################################################################################
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status

MOI.get(model::Optimizer, ::MOI.RawStatusString) = model.raw_status

"""
    MasterSolveCount() <: MOI.AbstractModelAttribute -> Int

Master MILP solves in the last `optimize!`, including the set-covering
pass and any pool-replacement re-solves.
"""
struct MasterSolveCount <: MOI.AbstractModelAttribute end

"""
    NLPSolveCount() <: MOI.AbstractModelAttribute -> Int

Indicator combinations evaluated by NLP subproblems in the last
`optimize!`; a stacked batch of `k` counts `k`.
"""
struct NLPSolveCount <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::Union{MasterSolveCount, NLPSolveCount}) = true
MOI.get(model::Optimizer, ::MasterSolveCount) = model.num_master_solves
MOI.get(model::Optimizer, ::NLPSolveCount) = model.num_nlp_solves

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.primal_status == MOI.NO_SOLUTION ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    return attr.result_index == 1 ? model.primal_status : MOI.NO_SOLUTION
end

MOI.get(model::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    bound = model.objective_bound
    if bound === nothing
        sense = MOI.get(model.cache, MOI.ObjectiveSense())
        return sense == MOI.MAX_SENSE ? Inf : -Inf
    end
    return bound
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex
    )
    MOI.check_result_index_bounds(model, attr)
    return model.incumbent[vi]
end

MOI.get(model::Optimizer, ::MOI.RelativeGap) = model.relative_gap

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
