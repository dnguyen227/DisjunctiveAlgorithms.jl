################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION
################################################################################
"""
    LOA()

`LOA` implements logic-based outer approximation (Turkay and
Grossmann 1996): a MILP master keeps the linear rows exactly, gates
each disjunct's rows on its indicator, and accumulates OA cuts from
NLP subproblems solved at fixed indicator combinations. The
set-covering pass of Turkay and Grossmann (1996, Appendix D),
restricted to the nonlinear disjuncts as in GDPopt (Chen, Johnson,
Siirola and Grossmann 2018), initializes it with combinations that
activate every nonlinear disjunct once; [`SetCoverProblem`](@ref)
picks the covering problem. The OA bound is only
valid on convex problems, so convergence reports `LOCALLY_SOLVED`,
never `OPTIMAL`.

## Supported optimizer attributes

- [`NumIterationLimit`](@ref)
- [`SetCoverIterationLimit`](@ref)
- [`SetCoverProblem`](@ref)
- [`M_Value`](@ref)
- [`MasterReformulation`](@ref)
- [`MaxSlack`](@ref)
- [`OASlack`](@ref)
- [`UseNLPF`](@ref)
- [`ConvergenceTolerance`](@ref)
- [`SlackTolerance`](@ref)
- [`IterationTimeLimit`](@ref)
- [`MultiGenerationSize`](@ref)
- [`CombinationSource`](@ref)
- [`SubproblemMethod`](@ref)
"""
mutable struct LOA <: AbstractAlgorithm
    num_iteration_limit::Union{Nothing, Int}
    set_cover_iteration_limit::Union{Nothing, Int}
    set_cover_problem::Union{Nothing, AbstractSetCover}
    m_value::Union{Nothing, Float64}
    master_reformulation::Union{Nothing, String}
    max_slack::Union{Nothing, Float64}
    oa_slack::Union{Nothing, Float64}
    use_nlpf::Union{Nothing, Bool}
    convergence_tolerance::Union{Nothing, Float64}
    slack_tolerance::Union{Nothing, Float64}
    iteration_time_limit::Union{Nothing, Float64}
    multi_generation_size::Union{Nothing, Int}
    combination_source::Any
    subproblem_method::Any

    LOA() = new(nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing)
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

function _validate(::SetCoverProblem, value)
    value isa AbstractSetCover || error("`SetCoverProblem` must be an " *
        "`AbstractSetCover`, e.g. `LogicOnlyCover()` (got `$value`).")
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
    (SetCoverProblem, :set_cover_problem),
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

# `nothing` is these attributes' default, so they skip the
# `something`-based fallback the loop above generates
MOI.supports(::LOA, ::SubproblemMethod) = true
function MOI.set(algorithm::LOA, ::SubproblemMethod, value)
    algorithm.subproblem_method = value
    return
end
MOI.get(algorithm::LOA, ::SubproblemMethod) = algorithm.subproblem_method

MOI.supports(::LOA, ::CombinationSource) = true
function MOI.set(algorithm::LOA, ::CombinationSource, value)
    algorithm.combination_source = value
    return
end
MOI.get(algorithm::LOA, ::CombinationSource) = algorithm.combination_source

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
    visited_disjunct_key = Set{Tuple{MOI.VariableIndex, Bool}}()
    cover = _Disjunct[]
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _is_nonlinear_disjunct(disjunct) || continue
        disjunct_key = (disjunct.binary, disjunct.active_value)
        disjunct_key in visited_disjunct_key && continue
        push!(visited_disjunct_key, disjunct_key)
        push!(cover, disjunct)
    end
    return cover
end

# The covering problem: a fresh logic-only model, or the master itself
_set_cover_model(::LogicOnlyCover, model, problem, master) =
    _build_set_cover(model, problem)
_set_cover_model(::MasterCover, model, problem, master) = master

# A covering solve on the master leaves the covering objective behind;
# the OA objective goes back before any cut is added
_restore_objective(::LogicOnlyCover, master::_Master) = nothing
_restore_objective(::MasterCover, master::_Master) =
    _set_master_objective(master, master.sense, master.oa_objective)

# The logic model needs its own no-good cut to move on; the master
# receives the combination's no-good cut from `apply_result`
_exclude_combination(::LogicOnlyCover, covering_model, combination) =
    _avoid_combination(covering_model, combination)
_exclude_combination(::MasterCover, covering_model, combination) = nothing

# Turkay and Grossmann's covering weights, as GDPopt uses them: an
# uncovered disjunct outweighs all covered ones
function _cover_objective(
    covering_model::Union{_Master, _SetCoverModel},
    cover::Vector{_Disjunct},
    needs_cover,
    num_covered::Int
    )
    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
    for i in eachindex(cover)
        weight = Float64(needs_cover[i] ? num_covered + 1 : 1)
        activation = _map_to(covering_model.variable_map, cover[i].activation)
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

function _solve_master(
    model::Optimizer,
    master::Union{_Master, _SetCoverModel},
    time_limit::Float64
    )
    _cap_remaining_time(master.model, time_limit)
    model.master_time += @elapsed MOI.optimize!(master.model)
    model.num_master_solves += 1
    return _solved_and_feasible(master.model)
end

# The master's proposal plus up to `count - 1` more from the
# `CombinationSource`. The second return value says whether the source
# already added the no-good cuts (re-solving sources must), so the
# caller skips them.
function _extract_combinations(
    combination_source,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    incumbent,
    count::Int,
    time_limit::Float64
    )
    master_combination = _extract_combination(problem, master)
    count > 1 || return [master_combination], false
    extra_candidates, cuts_added = _candidate_combinations(
        combination_source, model, problem, master, master_combination,
        incumbent, count - 1, time_limit)
    return [master_combination; extra_candidates], cuts_added
end

function _set_master_objective(
    master::Union{_Master, _SetCoverModel},
    sense,
    objective
    )
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
    taylor_cache = _Linearizer()
    sense = problem.sense
    overall_time_limit = t_start + something(model.time_limit_sec, Inf)
    loop_time_limit = min(overall_time_limit,
        t_start + Float64(MOI.get(algorithm, IterationTimeLimit())))

    best_objective = _worst_objective(sense)
    incumbent_result = nothing
    previous_result = _user_start_values(model, problem)
    master_bound = nothing
    master_status = nothing
    converged = false
    is_unbounded = false
    num_failed_nlps = 0

    # Shared iteration tail: no-good cut, OA cuts, incumbent update.
    # An unbounded NLP aborts the loops; a failed (neither feasible
    # nor proven-infeasible) NLP still gets its no-good cut so the
    # loop progresses, but is counted against the exhaustion claim.
    apply_result = (result; nogood = true) -> begin
        if !result.feasible && _nlp_unbounded(result.status)
            is_unbounded = true
            return
        end
        result.feasible || _nlp_infeasible(result.status) ||
            (num_failed_nlps += 1)
        result.feasible || (model.num_nlp_infeasible += 1)
        nogood && _avoid_combination(master, result.combination)
        _add_oa_cuts(model, problem, master, taylor_cache, result)
        if result.feasible &&
                _is_better(sense, result.objective, best_objective)
            best_objective = result.objective
            incumbent_result = result
        end
        result.feasible && (previous_result = result)
        _log_progress(model, t_start, best_objective, master_bound)
        return
    end
    warm_start = () ->
        previous_result === nothing ? nothing : previous_result.point

    # set covering: the covering problem (`SetCoverProblem`) picks
    # combinations that activate every nonlinear disjunct once; the
    # master receives their NLPs' cuts
    cover = _cover_disjuncts(problem)
    needs_cover = trues(length(cover))
    num_covered = 0
    covering_algorithm = MOI.get(algorithm, SetCoverProblem())
    covering_model = _set_cover_model(covering_algorithm, model, problem,
        master)
    for iteration in 1:MOI.get(algorithm, SetCoverIterationLimit())
        (iteration == 1 || any(needs_cover)) || break
        time() < loop_time_limit || break
        _set_master_objective(covering_model, MOI.MAX_SENSE,
            _cover_objective(covering_model, cover, needs_cover, num_covered))
        solved = _solve_master(model, covering_model, loop_time_limit)
        combination = solved ? _extract_combination(problem, covering_model) :
            nothing
        _restore_objective(covering_algorithm, master)
        solved || break
        # the remaining targets are unreachable once a solve covers none
        iteration == 1 || any(needs_cover[i] &&
            _disjunct_active(combination, cover[i])
            for i in eachindex(cover)) || break
        _exclude_combination(covering_algorithm, covering_model, combination)
        t_nlp = time()
        result = _solve_nlp(method, model, problem, subproblem,
            combination, warm_start(); time_limit = loop_time_limit)
        model.nlp_time += time() - t_nlp
        model.num_nlp_solves += 1
        apply_result(result)
        is_unbounded && break
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
    if master_status === nothing && !is_unbounded
        for _ in 1:MOI.get(algorithm, NumIterationLimit())
            time() < loop_time_limit || break
            if !_solve_master(model, master, loop_time_limit)
                master_status = MOI.get(master.model, MOI.TerminationStatus())
                break
            end
            master_bound = MOI.get(master.model, MOI.VariablePrimal(),
                master.alpha_oa)
            if incumbent_result !== nothing
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
            combinations, cuts_added = _extract_combinations(
                MOI.get(algorithm, CombinationSource()), model, problem,
                master, incumbent_result === nothing ? nothing :
                    incumbent_result.combination,
                MOI.get(algorithm, MultiGenerationSize()), loop_time_limit)
            t_nlp = time()
            results = _solve_nlps(method, model, problem, subproblem,
                combinations, warm_start(); time_limit = loop_time_limit)
            model.nlp_time += time() - t_nlp
            model.num_nlp_solves += length(results)
            for result in results
                apply_result(result; nogood = !cuts_added)
                is_unbounded && break
            end
            is_unbounded && break
        end
    end

    _store_results(model, sense, best_objective, incumbent_result,
        master_bound, master_status, converged, is_unbounded, num_failed_nlps,
        loop_time_limit)
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
    incumbent_result,
    master_bound,
    master_status,
    converged::Bool,
    is_unbounded::Bool,
    num_failed_nlps::Int,
    loop_time_limit::Float64
    )
    timed_out = time() >= loop_time_limit
    if is_unbounded
        model.termination_status = MOI.DUAL_INFEASIBLE
        model.primal_status = MOI.NO_SOLUTION
        model.objective_value = NaN
        model.raw_status = "An NLP subproblem at a fixed indicator " *
            "combination is unbounded, so the model is unbounded."
        return
    end
    if incumbent_result === nothing
        model.primal_status = MOI.NO_SOLUTION
        model.objective_value = NaN
        model.raw_status = "No feasible incumbent found."
        if master_status == MOI.INFEASIBLE && num_failed_nlps == 0
            model.termination_status = MOI.INFEASIBLE
            model.raw_status = "No feasible incumbent: the master " *
                "problem is infeasible."
        elseif master_status == MOI.INFEASIBLE
            # exhaustion is not an infeasibility proof while some
            # combination never solved to a certificate
            model.termination_status = MOI.OTHER_LIMIT
            model.raw_status = "No feasible incumbent: the master " *
                "is infeasible, but $num_failed_nlps combination(s) " *
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
    model.incumbent = incumbent_result.point
    model.objective_value = best_objective
    model.objective_bound = master_bound === nothing ? nothing :
        Float64(master_bound)
    combinations_exhausted = master_status == MOI.INFEASIBLE &&
        num_failed_nlps == 0
    if converged
        model.termination_status = MOI.LOCALLY_SOLVED
    elseif timed_out
        model.termination_status = MOI.TIME_LIMIT
    elseif combinations_exhausted
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
    outcome_label = converged ? "converged" :
        combinations_exhausted ? "combinations exhausted" :
        master_status == MOI.INFEASIBLE ? "combinations exhausted, " *
            "$num_failed_nlps unresolved" : "limit hit"
    if master_bound === nothing
        model.raw_status = "LOA finished [$outcome_label]: incumbent " *
            "$best_objective (master produced no bound)."
    else
        gap = _gap(sense, best_objective, master_bound)
        relative_gap = abs(best_objective) > 1e-10 ?
            gap / abs(best_objective) : gap
        model.relative_gap = relative_gap
        model.raw_status = "LOA finished [$outcome_label]: incumbent " *
            "$best_objective, master bound $master_bound, gap $gap " *
            "(relative $relative_gap)."
    end
    return
end
