################################################################################
#                          COMBINATION SOURCES
################################################################################
# Where the extra combinations of a multi-generation iteration come
# from. `_candidate_combinations(source, model, problem, master,
# proposal, incumbent, count, deadline)` returns up to `count`
# combinations other than `proposal` and whether it already added their
# no-good cuts to the master (a re-solving source must, so the loop
# skips them). The default `nothing` source is the solver's pool, then
# re-solves; the rest are experimental and need nothing from the solver.

const _Combination = Dict{MOI.VariableIndex, Bool}

# one chosen disjunct per disjunction -> combination
function _combination(problem::_Problem, chosen)
    combination = _Combination()
    for (disjunction, active) in zip(problem.disjunctions, chosen),
            disjunct in disjunction.disjuncts
        combination[disjunct.binary] = disjunct === active ?
            disjunct.active_value : !disjunct.active_value
    end
    return combination
end

# the active disjunct of each disjunction under a combination
function _active_disjuncts(problem::_Problem, combination::AbstractDict)
    return map(problem.disjunctions) do disjunction
        index = findfirst(d -> _disjunct_active(combination, d),
            disjunction.disjuncts)
        return disjunction.disjuncts[something(index, 1)]
    end
end

_push_candidate!(candidates, combination, proposal) =
    combination == proposal || combination in candidates ||
        push!(candidates, combination)

################################################################################
#                       DEFAULT: POOL, OR RE-SOLVES
################################################################################
# Result indices past 1 are the solver's pool (Gurobi's PoolSolutions);
# a short pool means a short batch. Only a solver without a pool falls
# back to re-solving behind a no-good cut per combination taken.
function _candidate_combinations(
    ::Nothing,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    incumbent,
    count::Int,
    deadline::Float64
    )
    candidates = _Combination[]
    for index in 2:min(count + 1, MOI.get(master.model, MOI.ResultCount()))
        _push_candidate!(candidates,
            _extract_combination(problem, master, index), proposal)
    end
    MOI.get(master.model, MOI.ResultCount()) > 1 && return candidates, false
    return _resolve_candidates(model, problem, master, proposal, candidates,
        count, deadline)
end

# Exclude the proposal and every candidate so far, then re-solve until
# `count` are in hand; the cuts stay, so the caller must not add them.
function _resolve_candidates(
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    candidates::Vector{_Combination},
    count::Int,
    deadline::Float64
    )
    _avoid_combination(master, proposal)
    foreach(c -> _avoid_combination(master, c), candidates)
    while length(candidates) < count && time() < deadline
        _solve_master(model, master, deadline) || break
        combination = _extract_combination(problem, master)
        combination == proposal && break
        combination in candidates && break
        push!(candidates, combination)
        _avoid_combination(master, combination)
    end
    return candidates, true
end

################################################################################
#                          CUTOFF RE-SOLVES
################################################################################
"""
    CutoffResolve(; tolerance = 0.05)

[`CombinationSource`](@ref) that re-solves the master behind no-good
cuts like the default, but with the master objective capped at the
current optimum plus `tolerance` (relative) and the last solution as a
warm start, so each re-solve is a small warm MILP. Still one master
solve per extra combination.
"""
struct CutoffResolve
    tolerance::Float64
    function CutoffResolve(; tolerance::Real = 0.05)
        tolerance >= 0 || error("`CutoffResolve` tolerance must be " *
            "nonnegative (got `$tolerance`).")
        return new(Float64(tolerance))
    end
end

function _candidate_combinations(
    source::CutoffResolve,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    incumbent,
    count::Int,
    deadline::Float64
    )
    # read the solution before any modification invalidates it
    value = MOI.get(master.model, MOI.ObjectiveValue())
    starts = _primal_starts(master)
    slack = source.tolerance * max(1.0, abs(value))
    set = master.sense == MOI.MAX_SENSE ? MOI.GreaterThan(value - slack) :
        MOI.LessThan(value + slack)
    cutoff = MOI.add_constraint(master.model, master.oa_objective, set)
    _set_primal_starts(master, starts)
    candidates, excluded = _resolve_candidates(model, problem, master,
        proposal, _Combination[], count, deadline)
    MOI.delete(master.model, cutoff)
    return candidates, excluded
end

function _primal_starts(master::_Master)
    MOI.supports(master.model, MOI.VariablePrimalStart(),
        MOI.VariableIndex) || return nothing
    return [vi => MOI.get(master.model, MOI.VariablePrimal(), vi)
        for vi in MOI.get(master.model, MOI.ListOfVariableIndices())]
end

function _set_primal_starts(master::_Master, starts)
    starts === nothing && return
    for (vi, value) in starts
        MOI.set(master.model, MOI.VariablePrimalStart(), vi, value)
    end
    return
end

################################################################################
#                            NEIGHBORHOOD
################################################################################
"""
    Neighborhood(; around = :proposal)

[`CombinationSource`](@ref) made of every combination that differs from
the master's proposal (or from the incumbent, `around = :incumbent`) in
exactly one disjunction's choice. Free: no solver call. A disjunction
of `m` disjuncts contributes `m - 1` neighbors; when there are more
than asked for, an evenly spaced subset over the disjunctions is taken.
"""
struct Neighborhood
    around::Symbol
    function Neighborhood(; around::Symbol = :proposal)
        around in (:proposal, :incumbent) || error("`Neighborhood` " *
            "`around` must be `:proposal` or `:incumbent` (got `$around`).")
        return new(around)
    end
end

function _candidate_combinations(
    source::Neighborhood,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    incumbent,
    count::Int,
    deadline::Float64
    )
    center = source.around == :incumbent && incumbent !== nothing ?
        incumbent : proposal
    chosen = _active_disjuncts(problem, center)
    candidates = _Combination[]
    for (i, disjunction) in enumerate(problem.disjunctions),
            disjunct in disjunction.disjuncts
        disjunct === chosen[i] && continue
        flipped = copy(chosen)
        flipped[i] = disjunct
        _push_candidate!(candidates, _combination(problem, flipped), proposal)
    end
    length(candidates) <= count && return candidates, false
    count == 1 && return candidates[1:1], false
    picks = round.(Int, range(1, length(candidates); length = count))
    return candidates[unique(picks)], false
end

################################################################################
#                        LP RELAXATION ROUNDING
################################################################################
"""
    LPRounding(; seed = 0)

[`CombinationSource`](@ref) that samples combinations from the master's
LP relaxation: integrality is dropped on a copy of the master, the LP
is solved, and each disjunction's active disjunct is drawn with
probability proportional to the relaxed value of its activation. One
LP per iteration, informed by every cut in the master. The LP solve is
not counted in `MasterSolveCount`.
"""
struct LPRounding
    rng::Random.MersenneTwister
    LPRounding(; seed::Integer = 0) = new(Random.MersenneTwister(seed))
end

function _candidate_combinations(
    source::LPRounding,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    incumbent,
    count::Int,
    deadline::Float64
    )
    weights = _relaxed_activations(model, problem, master, deadline)
    weights === nothing && return _Combination[], false
    candidates = _Combination[]
    for _ in 1:(20 * count)
        chosen = [_sample(source.rng, disjunction.disjuncts, w) for
            (disjunction, w) in zip(problem.disjunctions, weights)]
        _push_candidate!(candidates, _combination(problem, chosen), proposal)
        length(candidates) == count && break
    end
    return candidates, false
end

# relaxed activation value of every disjunct, per disjunction
function _relaxed_activations(
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    deadline::Float64
    )
    lp = _instantiate(model.mip_solver)
    index_map = MOI.copy_to(lp, master.model)
    for (F, S) in MOI.get(lp, MOI.ListOfConstraintTypesPresent())
        F === MOI.VariableIndex && S in (MOI.ZeroOne, MOI.Integer) || continue
        for ci in collect(MOI.get(lp, MOI.ListOfConstraintIndices{F, S}()))
            vi = MOI.get(lp, MOI.ConstraintFunction(), ci)
            MOI.delete(lp, ci)
            S === MOI.ZeroOne && _bound_unit_interval(lp, vi)
        end
    end
    _cap_remaining_time(lp, deadline)
    MOI.optimize!(lp)
    _solved_and_feasible(lp) || return nothing
    value = vi -> MOI.get(lp, MOI.VariablePrimal(),
        index_map[master.variable_map[vi]])
    return [[clamp(MOI.Utilities.eval_variables(value, disjunct.activation),
        0.0, 1.0) for disjunct in disjunction.disjuncts]
        for disjunction in problem.disjunctions]
end

# [0, 1] bounds for a relaxed binary, unless bounds already exist
function _bound_unit_interval(lp::MOI.ModelLike, vi::MOI.VariableIndex)
    interval = MOI.ConstraintIndex{MOI.VariableIndex,
        MOI.Interval{Float64}}(vi.value)
    MOI.is_valid(lp, interval) && return
    lower = MOI.ConstraintIndex{MOI.VariableIndex,
        MOI.GreaterThan{Float64}}(vi.value)
    MOI.is_valid(lp, lower) || MOI.add_constraint(lp, vi, MOI.GreaterThan(0.0))
    upper = MOI.ConstraintIndex{MOI.VariableIndex,
        MOI.LessThan{Float64}}(vi.value)
    MOI.is_valid(lp, upper) || MOI.add_constraint(lp, vi, MOI.LessThan(1.0))
    return
end

# weighted draw; uniform when the weights carry no information
function _sample(rng, disjuncts, weights)
    total = sum(weights)
    total > 0 || return rand(rng, disjuncts)
    threshold = rand(rng) * total
    running = 0.0
    for (disjunct, w) in zip(disjuncts, weights)
        running += w
        running >= threshold && return disjunct
    end
    return last(disjuncts)
end

################################################################################
#                          RANDOM COMBINATIONS
################################################################################
"""
    RandomCombinations(; seed = 0)

[`CombinationSource`](@ref) drawing combinations uniformly, one active
disjunct per disjunction. The uninformed baseline for the sources
above.
"""
struct RandomCombinations
    rng::Random.MersenneTwister
    RandomCombinations(; seed::Integer = 0) = new(Random.MersenneTwister(seed))
end

function _candidate_combinations(
    source::RandomCombinations,
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    proposal::AbstractDict,
    incumbent,
    count::Int,
    deadline::Float64
    )
    candidates = _Combination[]
    for _ in 1:(20 * count)
        chosen = [rand(source.rng, disjunction.disjuncts)
            for disjunction in problem.disjunctions]
        _push_candidate!(candidates, _combination(problem, chosen), proposal)
        length(candidates) == count && break
    end
    return candidates, false
end
