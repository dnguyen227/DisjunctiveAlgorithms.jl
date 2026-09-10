################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION
################################################################################
"""
    LOA()

`LOA` implements logic-based outer approximation (Turkay and
Grossmann 1996): a MILP master keeps the linear rows exactly, gates
each disjunct's rows on its indicator, and accumulates OA cuts from
NLP subproblems solved at fixed indicator combinations, seeded by a
set-covering pass over the nonlinear disjuncts. The OA bound is only
valid on convex problems, so convergence reports `LOCALLY_SOLVED`,
never `OPTIMAL`.

## Supported optimizer attributes

- [`NumIterationLimit`](@ref)
- [`SetCoverIterationLimit`](@ref)
- [`M_Value`](@ref)
- [`MasterReformulation`](@ref)
- [`MaxSlack`](@ref)
- [`OASlack`](@ref)
- [`UseNLPF`](@ref)
- [`ConvergenceTolerance`](@ref)
- [`SlackTolerance`](@ref)
- [`IterationTimeLimit`](@ref)
- [`MultiGenerationSize`](@ref)
- [`SubproblemMethod`](@ref)
"""
mutable struct LOA <: AbstractAlgorithm
    num_iteration_limit::Union{Nothing, Int}
    set_cover_iteration_limit::Union{Nothing, Int}
    m_value::Union{Nothing, Float64}
    master_reformulation::Union{Nothing, String}
    max_slack::Union{Nothing, Float64}
    oa_slack::Union{Nothing, Float64}
    use_nlpf::Union{Nothing, Bool}
    convergence_tolerance::Union{Nothing, Float64}
    slack_tolerance::Union{Nothing, Float64}
    iteration_time_limit::Union{Nothing, Float64}
    multi_generation_size::Union{Nothing, Int}
    subproblem_method::Any

    LOA() = new(nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing)
end

_default(::Algorithm) = LOA()

# Attribute values that would explode mid-solve fail at set time
# instead; attributes without a method to extend are unvalidated.
_validate(::AbstractAlgorithmAttribute, value) = nothing

function _validate(::MasterReformulation, value)
    value in ("indicator", "bigm") || error(
        "`MasterReformulation` must be \"indicator\" or \"bigm\" " *
        "(got `$value`).")
    return nothing
end

function _validate(::OASlack, value)
    value > 0 ||
        error("`OASlack` must be positive (got `$value`).")
    return nothing
end

function _validate(::MultiGenerationSize, value)
    value isa Integer && value >= 1 || error("`MultiGenerationSize` " *
        "must be a positive integer (got `$value`).")
    return nothing
end

# An unset field (`nothing`) reads back as the attribute default.
for (attr, field) in (
    (NumIterationLimit, :num_iteration_limit),
    (SetCoverIterationLimit, :set_cover_iteration_limit),
    (M_Value, :m_value),
    (MasterReformulation, :master_reformulation),
    (MaxSlack, :max_slack),
    (OASlack, :oa_slack),
    (UseNLPF, :use_nlpf),
    (ConvergenceTolerance, :convergence_tolerance),
    (SlackTolerance, :slack_tolerance),
    (IterationTimeLimit, :iteration_time_limit),
    (MultiGenerationSize, :multi_generation_size),
    )
    @eval begin
        MOI.supports(::LOA, ::$attr) = true
        function MOI.set(algorithm::LOA, attr::$attr, value)
            _validate(attr, value)
            algorithm.$field = value
            return
        end
        function MOI.get(algorithm::LOA, attr::$attr)
            return something(algorithm.$field, _default(algorithm, attr))
        end
    end
end

# `nothing` is this attribute's default, so it skips the
# `something`-based fallback the loop above generates
MOI.supports(::LOA, ::SubproblemMethod) = true
function MOI.set(algorithm::LOA, ::SubproblemMethod, value)
    algorithm.subproblem_method = value
    return
end
MOI.get(algorithm::LOA, ::SubproblemMethod) = algorithm.subproblem_method

_worst_objective(sense::MOI.OptimizationSense) =
    sense == MOI.MAX_SENSE ? -Inf : Inf
_is_better(sense::MOI.OptimizationSense, new, best) =
    sense == MOI.MAX_SENSE ? new > best : new < best
_gap(sense::MOI.OptimizationSense, best, bound) =
    sense == MOI.MAX_SENSE ? bound - best : best - bound

# The NLP fixes a feasible indicator combination, so an unbounded NLP
# proves the whole model unbounded; only a genuine infeasibility
# certificate lets a combination count toward exhaustion.
_nlp_unbounded(status::MOI.TerminationStatusCode) =
    status in (MOI.DUAL_INFEASIBLE, MOI.NORM_LIMIT)
_nlp_infeasible(status::MOI.TerminationStatusCode) =
    status in (MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE)

# one record per NLP solve, for convergence traces
function _log_progress(
    model::Optimizer,
    t_start::Float64,
    best_objective,
    master_bound
    )
    model.silent && return
    bound = master_bound === nothing ? NaN : master_bound
    @info "LOA progress: elapsed=$(time() - t_start) " *
        "incumbent=$best_objective bound=$bound"
    return
end

# one target per (binary, value); linear disjuncts need no cover
# since the master already carries them exactly
function _cover_disjuncts(problem::_Problem)
    seen = Set{Tuple{MOI.VariableIndex, Bool}}()
    cover = _Disjunct[]
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _is_nonlinear_disjunct(disjunct) || continue
        key = (disjunct.binary, disjunct.active_value)
        key in seen && continue
        push!(seen, key)
        push!(cover, disjunct)
    end
    return cover
end

# GDPopt's covering weights: an uncovered disjunct outweighs all
# covered ones
function _cover_objective(
    master::_Master,
    cover::Vector{_Disjunct},
    needs_cover,
    num_covered::Int
    )
    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
    for i in eachindex(cover)
        weight = Float64(needs_cover[i] ? num_covered + 1 : 1)
        activation = _map_to(master.variable_map, cover[i].activation)
        objective = MOI.Utilities.operate(+, Float64, objective,
            MOI.Utilities.operate(*, Float64, weight, activation))
    end
    return objective
end

# variable starts warm start the first NLP; later iterations use
# the last feasible primal
function _user_start_values(model::Optimizer, problem::_Problem)
    point = Dict{MOI.VariableIndex, Float64}()
    for vi in problem.variables
        start = MOI.get(model.cache, MOI.VariablePrimalStart(), vi)
        start === nothing || (point[vi] = start)
    end
    return isempty(point) ? nothing : (point = point,)
end

function _solve_master(model::Optimizer, master::_Master, deadline::Float64)
    _cap_remaining_time(master.model, deadline)
    MOI.optimize!(master.model)
    model.num_master_solves += 1
    return _solved_and_feasible(master.model)
end

# The master's solution pool first (result indices past 1), then
# re-solves behind a no-good cut for whatever the pool did not supply.
# Those cuts are the ones `process_result` would add later, so the
# second return value tells the caller to skip them.
function _extract_combinations(
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    count::Int,
    deadline::Float64
    )
    combinations = [_extract_combination(problem, master)]
    for index in 2:min(count, MOI.get(master.model, MOI.ResultCount()))
        combination = _extract_combination(problem, master, index)
        combination in combinations || push!(combinations, combination)
    end
    excluded = false
    while length(combinations) < count && time() < deadline
        if !excluded
            foreach(c -> _avoid_combination(master, c), combinations)
            excluded = true
        end
        _solve_master(model, master, deadline) || break
        combination = _extract_combination(problem, master)
        combination in combinations && break
        push!(combinations, combination)
        _avoid_combination(master, combination)
    end
    return combinations, excluded
end

function _set_master_objective(master::_Master, sense, objective)
    MOI.set(master.model, MOI.ObjectiveSense(), sense)
    MOI.set(master.model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        objective)
    return
end

################################################################################
#                              MAIN LOOP
################################################################################
function _optimize!(algorithm::LOA, model::Optimizer)
    t_start = time()
    _reset_results(model)
    problem = _build_problem(model)
    method = MOI.get(algorithm, SubproblemMethod())
    _check_inner_support(method, model, problem)
    master = _build_master(model, problem)
    subproblem = _build_subproblem(method, model, problem)
    linearizer = _Linearizer()
    sense = problem.sense
    overall_deadline = t_start + something(model.time_limit_sec, Inf)
    loop_deadline = min(overall_deadline,
        t_start + Float64(MOI.get(algorithm, IterationTimeLimit())))

    best_objective = _worst_objective(sense)
    best_result = nothing
    previous_result = _user_start_values(model, problem)
    master_bound = nothing
    master_status = nothing
    converged = false
    unbounded = false
    num_unresolved = 0

    # Shared iteration tail: no-good cut, OA cuts, incumbent update.
    # An unbounded NLP aborts the loops; a failed (neither feasible
    # nor proven-infeasible) NLP still gets its no-good cut so the
    # loop progresses, but is counted against the exhaustion claim.
    process_result = (result; nogood = true) -> begin
        if !result.feasible && _nlp_unbounded(result.status)
            unbounded = true
            return
        end
        result.feasible || _nlp_infeasible(result.status) ||
            (num_unresolved += 1)
        nogood && _avoid_combination(master, result.combination)
        _add_oa_cuts(model, problem, master, linearizer, result)
        if result.feasible &&
                _is_better(sense, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
        _log_progress(model, t_start, best_objective, master_bound)
        return
    end
    warm_start = () ->
        previous_result === nothing ? nothing : previous_result.point

    # set covering: reuse the master with a coverage objective so
    # every nonlinear disjunct gets visited once
    cover = _cover_disjuncts(problem)
    needs_cover = trues(length(cover))
    num_covered = 0
    for iteration in 1:MOI.get(algorithm, SetCoverIterationLimit())
        (iteration == 1 || any(needs_cover)) || break
        time() < loop_deadline || break
        _set_master_objective(master, MOI.MAX_SENSE,
            _cover_objective(master, cover, needs_cover, num_covered))
        solved = _solve_master(model, master, loop_deadline)
        # capture the status before the objective restore invalidates it
        status = MOI.get(master.model, MOI.TerminationStatus())
        combination = solved ? _extract_combination(problem, master) : nothing
        _set_master_objective(master, master.sense, master.oa_objective)
        if !solved
            master_status = status
            break
        end
        result = _solve_nlp(method, model, problem, subproblem,
            combination, warm_start(); deadline = loop_deadline)
        model.num_nlp_solves += 1
        process_result(result)
        unbounded && break
        # covered only once active in a feasible NLP; infeasible
        # combinations just leave their no-good cut
        if result.feasible
            for i in eachindex(cover)
                needs_cover[i] || continue
                _disjunct_active(result.combination, cover[i]) &&
                    (needs_cover[i] = false)
            end
            num_covered = count(!, needs_cover)
        end
    end

    # main loop: alpha_oa gives the bound, the NLP the incumbent
    if master_status === nothing && !unbounded
        for _ in 1:MOI.get(algorithm, NumIterationLimit())
            time() < loop_deadline || break
            if !_solve_master(model, master, loop_deadline)
                master_status = MOI.get(master.model, MOI.TerminationStatus())
                break
            end
            master_bound = MOI.get(master.model, MOI.VariablePrimal(),
                master.alpha_oa)
            if best_result !== nothing
                gap = _gap(sense, best_objective, master_bound)
                total_slack = abs(MOI.get(master.model,
                    MOI.ObjectiveValue()) - master_bound) /
                    Float64(MOI.get(algorithm, OASlack()))
                tol = Float64(MOI.get(algorithm, ConvergenceTolerance())) *
                    max(abs(best_objective), 1.0)
                if gap <= tol && total_slack <=
                        Float64(MOI.get(algorithm, SlackTolerance()))
                    converged = true
                    break
                end
            end
            combinations, excluded = _extract_combinations(model, problem,
                master, MOI.get(algorithm, MultiGenerationSize()),
                loop_deadline)
            results = _solve_nlps(method, model, problem, subproblem,
                combinations, warm_start(); deadline = loop_deadline)
            model.num_nlp_solves += length(results)
            for result in results
                process_result(result; nogood = !excluded)
                unbounded && break
            end
            unbounded && break
        end
    end

    _store_results(model, sense, best_objective, best_result, master_bound,
        master_status, converged, unbounded, num_unresolved, loop_deadline)
    model.solve_time = time() - t_start
    return
end

################################################################################
#                          RESULT SYNTHESIS
################################################################################
# the OA bound is valid only for convex problems, so report
# LOCALLY_SOLVED, never OPTIMAL
function _store_results(
    model::Optimizer,
    sense::MOI.OptimizationSense,
    best_objective::Float64,
    best_result,
    master_bound,
    master_status,
    converged::Bool,
    unbounded::Bool,
    num_unresolved::Int,
    loop_deadline::Float64
    )
    timed_out = time() >= loop_deadline
    if unbounded
        model.termination_status = MOI.DUAL_INFEASIBLE
        model.primal_status = MOI.NO_SOLUTION
        model.objective_value = NaN
        model.raw_status = "An NLP subproblem at a fixed indicator " *
            "combination is unbounded, so the model is unbounded."
        return
    end
    if best_result === nothing
        model.primal_status = MOI.NO_SOLUTION
        model.objective_value = NaN
        model.raw_status = "No feasible incumbent found."
        if master_status == MOI.INFEASIBLE && num_unresolved == 0
            model.termination_status = MOI.INFEASIBLE
            model.raw_status = "No feasible incumbent: the master " *
                "problem is infeasible."
        elseif master_status == MOI.INFEASIBLE
            # exhaustion is not an infeasibility proof while some
            # combination never solved to a certificate
            model.termination_status = MOI.OTHER_LIMIT
            model.raw_status = "No feasible incumbent: the master " *
                "is infeasible, but $num_unresolved combination(s) " *
                "failed without an infeasibility certificate."
        elseif timed_out
            model.termination_status = MOI.TIME_LIMIT
        elseif master_status === nothing
            model.termination_status = MOI.ITERATION_LIMIT
        elseif master_status == MOI.DUAL_INFEASIBLE
            model.termination_status = MOI.OTHER_LIMIT
            model.raw_status = "No feasible incumbent: the master is " *
                "unbounded (no OA cut bounds the objective yet); the " *
                "model itself may be unbounded."
        else
            model.termination_status = MOI.OTHER_LIMIT
            model.raw_status = "No feasible incumbent: the master " *
                "solve finished with status $master_status."
        end
        return
    end
    model.primal_status = MOI.FEASIBLE_POINT
    model.incumbent = best_result.point
    model.objective_value = best_objective
    model.objective_bound = master_bound === nothing ? nothing :
        Float64(master_bound)
    exhausted = master_status == MOI.INFEASIBLE && num_unresolved == 0
    if converged
        model.termination_status = MOI.LOCALLY_SOLVED
    elseif timed_out
        model.termination_status = MOI.TIME_LIMIT
    elseif exhausted
        # all combinations visited; the incumbent is best over all
        # of them, but the OA bound is gone
        model.termination_status = MOI.LOCALLY_SOLVED
    elseif master_status == MOI.INFEASIBLE
        # visited, but some combination failed without a certificate,
        # so the incumbent is not best over all of them
        model.termination_status = MOI.OTHER_LIMIT
    elseif master_status !== nothing
        model.termination_status = MOI.OTHER_LIMIT
    else
        model.termination_status = MOI.ITERATION_LIMIT
    end
    label = converged ? "converged" :
        exhausted ? "combinations exhausted" :
        master_status == MOI.INFEASIBLE ? "combinations exhausted, " *
            "$num_unresolved unresolved" : "limit hit"
    if master_bound === nothing
        model.raw_status = "LOA finished [$label]: incumbent " *
            "$best_objective (master produced no bound)."
    else
        gap = _gap(sense, best_objective, master_bound)
        relative = abs(best_objective) > 1e-10 ?
            gap / abs(best_objective) : gap
        model.relative_gap = relative
        model.raw_status = "LOA finished [$label]: incumbent " *
            "$best_objective, master bound $master_bound, gap $gap " *
            "(relative $relative)."
    end
    return
end
