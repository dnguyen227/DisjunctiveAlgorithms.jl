################################################################################
#                          NLP SUBPROBLEM
################################################################################
# The MOI subproblem path dispatches on the `SubproblemMethod` value
# `nothing`; a method object routes `_build_subproblem`/`_solve_nlp`
# to its own construction (see the `SubproblemMethod` docstring).
# Built once; each iteration overwrites the binary fixes in place
# and swaps the active disjuncts' rows. No big-M anywhere.
struct _Subproblem
    model::MOI.ModelLike
    variable_map::Dict{MOI.VariableIndex, MOI.VariableIndex}
    fixes::Dict{MOI.VariableIndex,
        MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}
    rows::Vector{MOI.ConstraintIndex}
end

# One copy of the variables, bounds, binary fixes (at 0 until a
# combination is set), and global rows; disjunct rows depend on the
# combination and are the caller's.
function _add_copy(nlp::MOI.ModelLike, model::Optimizer, problem::_Problem)
    variable_map = Dict{MOI.VariableIndex, MOI.VariableIndex}(
        vi => MOI.add_variable(nlp) for vi in problem.variables)
    indicators = Set(problem.binaries)
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        vi in indicators && continue
        MOI.add_constraint(nlp, variable_map[vi],
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    fixes = Dict(binary => MOI.add_constraint(nlp, variable_map[binary],
        MOI.EqualTo(0.0)) for binary in problem.binaries)
    for ci in problem.linear_cis
        func = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        MOI.add_constraint(nlp, _map_to(variable_map, func),
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    for (func, set) in problem.nonlinear_rows
        MOI.add_constraint(nlp, _map_to(variable_map, func), set)
    end
    return variable_map, fixes
end

# Point one copy at a combination: overwrite its binary fixes and add
# the active disjuncts' rows, returned so a reused copy can drop them.
function _fix_combination(
    nlp::MOI.ModelLike,
    variable_map::AbstractDict,
    fixes::AbstractDict,
    problem::_Problem,
    combination::AbstractDict
    )
    for (binary, value) in combination
        MOI.set(nlp, MOI.ConstraintSet(), fixes[binary],
            MOI.EqualTo(value ? 1.0 : 0.0))
    end
    rows = MOI.ConstraintIndex[]
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _disjunct_active(combination, disjunct) || continue
        for (func, set) in zip(disjunct.functions, disjunct.sets)
            push!(rows, MOI.add_constraint(nlp,
                _map_to(variable_map, func), set))
        end
    end
    return rows
end

function _build_subproblem(::Nothing, model::Optimizer, problem::_Problem)
    nlp = _instantiate(model.nlp_solver)
    variable_map, fixes = _add_copy(nlp, model, problem)
    MOI.set(nlp, MOI.ObjectiveSense(), problem.sense)
    objective = _map_to(variable_map, problem.objective)
    MOI.set(nlp, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    return _Subproblem(nlp, variable_map, fixes, MOI.ConstraintIndex[])
end

function _set_warm_start(nlp::MOI.ModelLike, variable_map::AbstractDict, point)
    point === nothing && return
    for (vi, value) in point
        MOI.set(nlp, MOI.VariablePrimalStart(), variable_map[vi], value)
    end
    return
end

function _extract_point(
    nlp::MOI.ModelLike,
    problem::_Problem,
    variable_map::AbstractDict
    )
    return Dict{MOI.VariableIndex, Float64}(
        vi => MOI.get(nlp, MOI.VariablePrimal(), variable_map[vi])
        for vi in problem.variables)
end

# Solve the NLP at a fixed combination: overwrite the binary fixes,
# swap the active disjuncts' rows, and optimize. If infeasible, fall
# through to NLPF (a slacked version that always solves) so the master
# still gets a linearization site, not just a no-good cut.
function _solve_nlp(
    ::Nothing,
    model::Optimizer,
    problem::_Problem,
    sub::_Subproblem,
    combination::AbstractDict,
    warm_start;
    deadline::Float64 = Inf
    )
    for ci in sub.rows
        MOI.delete(sub.model, ci)
    end
    append!(empty!(sub.rows), _fix_combination(sub.model, sub.variable_map,
        sub.fixes, problem, combination))
    _set_warm_start(sub.model, sub.variable_map, warm_start)
    _cap_remaining_time(sub.model, deadline)
    MOI.optimize!(sub.model)
    status = MOI.get(sub.model, MOI.TerminationStatus())
    if _solved_and_feasible(sub.model)
        return (combination = combination,
            point = _extract_point(sub.model, problem, sub.variable_map),
            objective = MOI.get(sub.model, MOI.ObjectiveValue()),
            feasible = true, status = status)
    end
    if Bool(MOI.get(model, UseNLPF()))
        result = _solve_nlpf(model, problem, combination, warm_start;
            deadline = deadline)
        result === nothing || return (; result..., status = status)
    end
    return (combination = combination,
        point = nothing, objective = Inf, feasible = false, status = status)
end

# Several combinations at once; the default method has no batch form
# and takes them one at a time.
function _solve_nlps(
    method,
    model::Optimizer,
    problem::_Problem,
    sub,
    combinations::AbstractVector,
    warm_start;
    deadline::Float64 = Inf
    )
    return [_solve_nlp(method, model, problem, sub, combination, warm_start;
        deadline) for combination in combinations]
end

################################################################################
#                        BATCHED SUBPROBLEMS
################################################################################
"""
    BatchedSubproblems()

`SubproblemMethod` value that solves a vector of indicator combinations
as one block-diagonal NLP through the `nlp_solver`: each combination
gets its own copy of the variables and rows, and the objective is the
sum of the copies' objectives. The copies do not interact, so the
stacked solution is the sequential one; what changes is that the
solver sees one large, regular problem, which is what GPU evaluators
and factorizations need to pay off. A single combination is a batch of
one. A batch whose joint solve fails (one infeasible copy makes the
whole stack infeasible) falls back to the sequential path copy by copy,
so the results match the default method exactly.
"""
struct BatchedSubproblems end

# the sequential subproblem is the fallback for failed batches
struct _BatchedSubproblem
    sequential::_Subproblem
end

_check_nlp_support(::BatchedSubproblems, model::Optimizer, problem::_Problem) =
    _check_nlp_support(nothing, model, problem)

function _build_subproblem(
    ::BatchedSubproblems,
    model::Optimizer,
    problem::_Problem
    )
    return _BatchedSubproblem(_build_subproblem(nothing, model, problem))
end

function _solve_nlp(
    method::BatchedSubproblems,
    model::Optimizer,
    problem::_Problem,
    sub::_BatchedSubproblem,
    combination::AbstractDict,
    warm_start;
    deadline::Float64 = Inf
    )
    return only(_solve_nlps(method, model, problem, sub, [combination],
        warm_start; deadline))
end

# One stacked solve of the combinations as given; `nothing` when the
# joint problem did not solve to a feasible point.
function _solve_stack(
    model::Optimizer,
    problem::_Problem,
    combinations::AbstractVector,
    warm_start,
    deadline::Float64
    )
    nlp = _instantiate(model.nlp_solver)
    copies = map(combinations) do combination
        variable_map, fixes = _add_copy(nlp, model, problem)
        _fix_combination(nlp, variable_map, fixes, problem, combination)
        _set_warm_start(nlp, variable_map, warm_start)
        return variable_map
    end
    objective = foldl((a, b) -> _combine(+, a, b),
        (_map_to(variable_map, problem.objective) for variable_map in copies))
    MOI.set(nlp, MOI.ObjectiveSense(), problem.sense)
    MOI.set(nlp, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    _cap_remaining_time(nlp, deadline)
    MOI.optimize!(nlp)
    _solved_and_feasible(nlp) || return nothing
    status = MOI.get(nlp, MOI.TerminationStatus())
    return map(zip(combinations, copies)) do (combination, variable_map)
        point = _extract_point(nlp, problem, variable_map)
        objective = MOI.Utilities.eval_variables(vi -> point[vi],
            model.cache, problem.objective)
        return (combination = combination, point = point,
            objective = objective, feasible = true, status = status)
    end
end

# Stacked feasibility pass: every copy slacked, sum of slacks minimized.
# Copies with zero slack are feasible; the others' points are their
# restoration points. `nothing` when even the slacked stack fails.
function _solve_slacked_stack(
    model::Optimizer,
    problem::_Problem,
    combinations::AbstractVector,
    warm_start,
    deadline::Float64
    )
    nlp = _instantiate(model.nlp_solver)
    copies = map(combinations) do combination
        variable_map, u = _add_slacked_copy(nlp, model, problem, combination)
        _set_warm_start(nlp, variable_map, warm_start)
        return variable_map, u
    end
    MOI.set(nlp, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(nlp, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, u) for (_, u) in copies], 0.0))
    _cap_remaining_time(nlp, deadline)
    MOI.optimize!(nlp)
    _solved_and_feasible(nlp) || return nothing
    tolerance = Float64(MOI.get(model, SlackTolerance()))
    feasible = [MOI.get(nlp, MOI.VariablePrimal(), u) <= tolerance
        for (_, u) in copies]
    points = [_extract_point(nlp, problem, variable_map)
        for (variable_map, _) in copies]
    return feasible, points
end

# Stack first; if one copy is infeasible the stack is, so classify the
# copies with a stacked feasibility pass and re-stack the feasible
# ones. Sequential solves remain only for a stack that fails on its
# own account.
function _solve_nlps(
    ::BatchedSubproblems,
    model::Optimizer,
    problem::_Problem,
    sub::_BatchedSubproblem,
    combinations::AbstractVector,
    warm_start;
    deadline::Float64 = Inf
    )
    sequential = () -> _solve_nlps(nothing, model, problem, sub.sequential,
        combinations, warm_start; deadline)
    results = _solve_stack(model, problem, combinations, warm_start, deadline)
    results === nothing || return results
    Bool(MOI.get(model, UseNLPF())) || return sequential()
    restored = _solve_slacked_stack(model, problem, combinations, warm_start,
        deadline)
    restored === nothing && return sequential()
    feasible, points = restored
    results = Vector{Any}(undef, length(combinations))
    active = findall(feasible)
    if !isempty(active)
        solved = _solve_stack(model, problem, combinations[active],
            warm_start, deadline)
        solved === nothing && (solved = _solve_nlps(nothing, model, problem,
            sub.sequential, combinations[active], warm_start; deadline))
        results[active] .= solved
    end
    for i in findall(!, feasible)
        results[i] = (combination = combinations[i], point = points[i],
            objective = Inf, feasible = false,
            status = MOI.LOCALLY_INFEASIBLE)
    end
    return results
end

################################################################################
#                       NLPF (FEASIBILITY SUBPROBLEM)
################################################################################
_nlpf_slacked(func, u, ::MOI.LessThan{Float64}) =
    MOI.Utilities.operate(-, Float64, func, u)
_nlpf_slacked(func, u, ::MOI.GreaterThan{Float64}) =
    MOI.Utilities.operate(+, Float64, func, u)
_nlpf_slacked(func, u, ::MOI.AbstractScalarSet) = nothing

# The slacked feasibility NLP: one nonnegative `u` relaxes every scalar
# inequality row (bounds and equalities stay exact) and is minimized.
# Its solution is a linearization site for an infeasible combination.
# One slacked copy at a combination: bounds and binary fixes exact,
# every inequality row relaxed by the copy's nonnegative `u`.
function _add_slacked_copy(
    nlp::MOI.ModelLike,
    model::Optimizer,
    problem::_Problem,
    combination::AbstractDict
    )
    variable_map = Dict{MOI.VariableIndex, MOI.VariableIndex}(
        vi => MOI.add_variable(nlp) for vi in problem.variables)
    u = MOI.add_variable(nlp)
    MOI.add_constraint(nlp, u, MOI.GreaterThan(0.0))
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        haskey(combination, vi) && continue
        MOI.add_constraint(nlp, variable_map[vi],
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    for (binary, value) in combination
        MOI.add_constraint(nlp, variable_map[binary],
            MOI.EqualTo(value ? 1.0 : 0.0))
    end
    rows = Tuple{MOI.AbstractScalarFunction, MOI.AbstractScalarSet}[]
    for ci in problem.linear_cis
        push!(rows, (MOI.get(model.cache, MOI.ConstraintFunction(), ci),
            MOI.get(model.cache, MOI.ConstraintSet(), ci)))
    end
    append!(rows, problem.nonlinear_rows)
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _disjunct_active(combination, disjunct) || continue
        append!(rows, zip(disjunct.functions, disjunct.sets))
    end
    for (func, set) in rows
        mapped = _map_to(variable_map, func)
        slacked = _nlpf_slacked(mapped, u, set)
        MOI.add_constraint(nlp, something(slacked, mapped), set)
    end
    return variable_map, u
end

function _solve_nlpf(
    model::Optimizer,
    problem::_Problem,
    combination::AbstractDict,
    warm_start;
    deadline::Float64 = Inf
    )
    nlp = _instantiate(model.nlp_solver)
    variable_map, u = _add_slacked_copy(nlp, model, problem, combination)
    MOI.set(nlp, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(nlp, MOI.ObjectiveFunction{MOI.VariableIndex}(), u)
    _set_warm_start(nlp, variable_map, warm_start)
    _cap_remaining_time(nlp, deadline)
    MOI.optimize!(nlp)
    # Use the primal only at a genuine feasible point; a solver can
    # report values at a nonfeasible/NaN primal that hurts the cut.
    _solved_and_feasible(nlp) || return nothing
    return (combination = combination,
        point = _extract_point(nlp, problem, variable_map),
        objective = Inf, feasible = false)
end
