module DisjunctiveAlgorithmsExaModelsExt

import DisjunctiveAlgorithms as DA
import ExaModels
import MathOptInterface as MOI

# ExaModels' MOI wrapper (its Optimizer and solver-status table); looked
# up at use, since it may load after this extension
_exa_moi() = Base.get_extension(ExaModels, :ExaModelsMOI)

# One ExaModel, host copies of its bounds and start, and the model column
# of each problem variable
struct _ExaCopy
    exa_model::Any
    column::Dict{MOI.VariableIndex, Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    start::Vector{Float64}
end

# The NLP and, with UseNLPF, its feasibility NLP, each built once
struct _ExaSubproblem
    optimality::_ExaCopy
    feasibility::Union{_ExaCopy, Nothing}
    solver::Any
    options::Dict{Symbol, Any}
end

function DA._check_nlp_support(
    ::DA.ExaModelSubproblems,
    model::DA.Optimizer,
    problem::DA._Problem
    )
    DA._check_nlp_support(nothing, model, problem)
    MOI.instantiate(model.nlp_solver) isa _exa_moi().Optimizer || error(
        "`ExaModelSubproblems` needs an `ExaModels.Optimizer` as the " *
        "`nlp_solver`.")
    all(isempty(disjunct.functions) for disjunction in problem.disjunctions
        for disjunct in disjunction.disjuncts) || error(
        "`ExaModelSubproblems` keeps one model for every combination, so " *
        "every disjunct must be empty: write each disjunct row as a " *
        "global row gated by its indicator binary.")
    return
end

# shortcut: ExaModel(::MOI.ModelLike) adds variables in
# ListOfVariableIndices order and returns no index map; build the
# ExaCore here if that order ever stops being the column order
function _exa_copy(source, variable_map, problem::DA._Problem, backend)
    exa_model = ExaModels.ExaModel(source; backend = backend, prod = true)
    order = MOI.get(source, MOI.ListOfVariableIndices())
    position = Dict(vi => k for (k, vi) in enumerate(order))
    column = Dict(vi => position[variable_map[vi]] for vi in problem.variables)
    meta = exa_model.meta
    return _ExaCopy(exa_model, column, Array(meta.lvar), Array(meta.uvar),
        Array(meta.x0))
end

function DA._build_subproblem(
    ::DA.ExaModelSubproblems,
    model::DA.Optimizer,
    problem::DA._Problem
    )
    wrapper = MOI.instantiate(model.nlp_solver)
    # the default method's NLP (variables, bounds, binary fixes, global
    # rows, objective) is the source of the NLP's ExaModel
    source = DA._build_subproblem(nothing, model, problem)
    optimality = _exa_copy(source.model, source.variable_map, problem,
        wrapper.backend)
    feasibility = nothing
    if Bool(MOI.get(model, DA.UseNLPF()))
        # the NLPF's slacked rows at an all-false combination; the binary
        # fixes are rewritten per combination as on the NLP
        nlp = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
        variable_map, slack = DA._add_slacked_copy(nlp, model, problem,
            Dict(binary => false for binary in problem.binaries))
        MOI.set(nlp, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(nlp, MOI.ObjectiveFunction{MOI.VariableIndex}(), slack)
        feasibility = _exa_copy(nlp, variable_map, problem, wrapper.backend)
    end
    return _ExaSubproblem(optimality, feasibility, wrapper.solver,
        wrapper.options)
end

# Point one ExaModel at a combination and a start, solve it, and return
# its MOI status and the solver's result
function _solve_copy(sub::_ExaSubproblem, exa_copy::_ExaCopy, combination,
        warm_start)
    lower, upper = copy(exa_copy.lower), copy(exa_copy.upper)
    start = copy(exa_copy.start)
    if warm_start !== nothing
        for (vi, value) in warm_start
            haskey(exa_copy.column, vi) && (start[exa_copy.column[vi]] = value)
        end
    end
    for (binary, value) in combination
        column = exa_copy.column[binary]
        lower[column] = upper[column] = start[column] = value ? 1.0 : 0.0
    end
    meta = exa_copy.exa_model.meta
    copyto!(meta.lvar, lower)
    copyto!(meta.uvar, upper)
    copyto!(meta.x0, start)
    result = sub.solver(exa_copy.exa_model; sub.options...)
    exa_moi = _exa_moi()
    status = get(exa_moi._TERMINATION_STATUS_CODES,
        exa_moi._status_symbol(result.status), MOI.OTHER_ERROR)
    return status, result
end

function _point(exa_copy::_ExaCopy, result, problem::DA._Problem)
    solution = Array(result.solution)
    return Dict(vi => solution[exa_copy.column[vi]] for vi in problem.variables)
end

# shortcut: DA's remaining time is not passed to the solver; its own
# options (e.g. MadNLP's max_wall_time) bound each solve
function DA._solve_nlp(
    ::DA.ExaModelSubproblems,
    model::DA.Optimizer,
    problem::DA._Problem,
    sub::_ExaSubproblem,
    combination::AbstractDict,
    warm_start;
    time_limit::Float64 = Inf
    )
    status, result = _solve_copy(sub, sub.optimality, combination, warm_start)
    status == MOI.LOCALLY_SOLVED && return (combination = combination,
        point = _point(sub.optimality, result, problem),
        objective = result.objective, feasible = true, status = status)
    if sub.feasibility !== nothing
        # a restoration point only from a solved NLPF, as the default
        # method's
        nlpf_status, nlpf_result = _solve_copy(sub, sub.feasibility,
            combination, warm_start)
        nlpf_status == MOI.LOCALLY_SOLVED && return (combination = combination,
            point = _point(sub.feasibility, nlpf_result, problem),
            objective = Inf, feasible = false, status = status)
    end
    return (combination = combination, point = nothing, objective = Inf,
        feasible = false, status = status)
end

end
