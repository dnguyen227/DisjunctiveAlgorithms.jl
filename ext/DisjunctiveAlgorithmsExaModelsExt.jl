module DisjunctiveAlgorithmsExaModelsExt

import DisjunctiveAlgorithms as DA
import ExaModels
import MathOptInterface as MOI

# ExaModels' MOI wrapper (its Optimizer and solver-status table); looked
# up at use, since it may load after this extension
_exa_moi() = Base.get_extension(ExaModels, :ExaModelsMOI)

# One ExaModel for every combination; host copies of its bounds and
# start, and the model column of each problem variable
struct _ExaSubproblem
    exa_model::Any
    solver::Any
    options::Dict{Symbol, Any}
    column::Dict{MOI.VariableIndex, Int}
    lower::Vector{Float64}
    upper::Vector{Float64}
    start::Vector{Float64}
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

function DA._build_subproblem(
    ::DA.ExaModelSubproblems,
    model::DA.Optimizer,
    problem::DA._Problem
    )
    # the default method's NLP (variables, bounds, binary fixes, global
    # rows, objective) is the source the ExaModel is built from
    source = DA._build_subproblem(nothing, model, problem)
    wrapper = MOI.instantiate(model.nlp_solver)
    exa_model = ExaModels.ExaModel(source.model; backend = wrapper.backend,
        prod = true)
    # shortcut: ExaModel(::MOI.ModelLike) adds variables in
    # ListOfVariableIndices order and returns no index map; build the
    # ExaCore here if that order ever stops being the column order
    order = MOI.get(source.model, MOI.ListOfVariableIndices())
    position = Dict(vi => k for (k, vi) in enumerate(order))
    column = Dict(vi => position[source.variable_map[vi]]
        for vi in problem.variables)
    meta = exa_model.meta
    return _ExaSubproblem(exa_model, wrapper.solver, wrapper.options, column,
        Array(meta.lvar), Array(meta.uvar), Array(meta.x0))
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
    lower, upper, start = copy(sub.lower), copy(sub.upper), copy(sub.start)
    if warm_start !== nothing
        for (vi, value) in warm_start
            haskey(sub.column, vi) && (start[sub.column[vi]] = value)
        end
    end
    for (binary, value) in combination
        column = sub.column[binary]
        lower[column] = upper[column] = start[column] = value ? 1.0 : 0.0
    end
    meta = sub.exa_model.meta
    copyto!(meta.lvar, lower)
    copyto!(meta.uvar, upper)
    copyto!(meta.x0, start)
    result = sub.solver(sub.exa_model; sub.options...)
    exa_moi = _exa_moi()
    status = get(exa_moi._TERMINATION_STATUS_CODES,
        exa_moi._status_symbol(result.status), MOI.OTHER_ERROR)
    if status == MOI.LOCALLY_SOLVED
        solution = Array(result.solution)
        point = Dict(vi => solution[sub.column[vi]] for vi in problem.variables)
        return (combination = combination, point = point,
            objective = result.objective, feasible = true, status = status)
    end
    if Bool(MOI.get(model, DA.UseNLPF()))
        nlpf = DA._solve_nlpf(model, problem, combination, warm_start;
            time_limit = time_limit)
        nlpf === nothing || return (; nlpf..., status = status)
    end
    return (combination = combination, point = nothing, objective = Inf,
        feasible = false, status = status)
end

end
