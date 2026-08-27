################################################################################
#                          PROBLEM PARTITION
################################################################################
# activation is `z` or `1 - z`; rows stay in cache space
struct _Disjunct
    activation::MOI.ScalarAffineFunction{Float64}
    binary::MOI.VariableIndex
    active_value::Bool
    functions::Vector{MOI.AbstractScalarFunction}
    sets::Vector{MOI.AbstractScalarSet}
end

# activation: 1 at top level, the parent indicator when nested
struct _Disjunction
    activation::MOI.ScalarAffineFunction{Float64}
    disjuncts::Vector{_Disjunct}
end

# Constraint indices stay in cache space; the builders remap them
struct _Problem
    variables::Vector{MOI.VariableIndex}
    binaries::Vector{MOI.VariableIndex}
    disjunctions::Vector{_Disjunction}
    variable_cis::Vector{MOI.ConstraintIndex}
    linear_cis::Vector{MOI.ConstraintIndex}
    # stable function objects so the linearizer can cache evaluators
    nonlinear_rows::Vector{Tuple{MOI.AbstractScalarFunction,
        MOI.AbstractScalarSet}}
    sense::MOI.OptimizationSense
    objective::MOI.AbstractScalarFunction
end

_is_linear(::Union{MOI.VariableIndex, MOI.ScalarAffineFunction}) = true
_is_linear(::MOI.AbstractScalarFunction) = false

_to_affine(func::MOI.ScalarAffineFunction{Float64}) = func
function _to_affine(func::MOI.AbstractScalarFunction)
    return convert(MOI.ScalarAffineFunction{Float64}, func)
end

# a raw `VariableIndex` row becomes a bound and collides with the
# variable's own bounds in the subproblem
_as_row(func::MOI.VariableIndex) = _to_affine(func)
_as_row(func::MOI.AbstractScalarFunction) = func

# Polynomial degree of an operator tree built from +, -, * and ^;
# `nothing` for any other operator. Promoted affine and quadratic rows
# only use these, so the degree says which MOI type they convert to.
_degree(::Real) = 0
_degree(::MOI.VariableIndex) = 1
_degree(::MOI.ScalarAffineFunction) = 1
_degree(::MOI.ScalarQuadraticFunction) = 2
function _degree(func::MOI.ScalarNonlinearFunction)
    degrees = [_degree(arg) for arg in func.args]
    any(isnothing, degrees) && return nothing
    func.head in (:+, :-) && return maximum(degrees)
    func.head == :* && return sum(degrees)
    func.head == :^ && func.args[2] isa Real &&
        return degrees[1] * func.args[2]
    return nothing
end

# demote rows the enclosing vector function promoted
_demote(func::MOI.AbstractScalarFunction) = func
function _demote(func::MOI.ScalarQuadraticFunction{Float64})
    return isempty(func.quadratic_terms) ? _to_affine(func) : func
end
function _demote(func::MOI.ScalarNonlinearFunction)
    degree = _degree(func)
    degree in (0, 1) && return _to_affine(func)
    degree == 2 && return convert(MOI.ScalarQuadraticFunction{Float64}, func)
    return func
end

_scalarize(func::MOI.AbstractVectorFunction) =
    collect(MOI.Utilities.eachscalar(func))

function _indicator_error(func::MOI.AbstractScalarFunction)
    return error("Unsupported indicator expression `$func`: each " *
        "`DisjunctionSet` indicator must be a binary variable `z` or " *
        "its complement `1 - z`.")
end

# An activation/indicator row must demote to an affine expression;
# genuinely nonlinear rows get the curated error, not a raw convert
# failure.
function _activation_affine(row::MOI.AbstractScalarFunction)
    demoted = _demote(row)
    _is_linear(demoted) || _indicator_error(demoted)
    return _to_affine(demoted)
end

# `z` -> (z, true), `1 - z` -> (z, false)
function _activation_binary(activation::MOI.ScalarAffineFunction{Float64})
    canonical = MOI.Utilities.canonical(activation)
    if length(canonical.terms) == 1
        term = only(canonical.terms)
        if term.coefficient == 1.0 && canonical.constant == 0.0
            return term.variable, true
        elseif term.coefficient == -1.0 && canonical.constant == 1.0
            return term.variable, false
        end
    end
    return _indicator_error(activation)
end

function _parse_disjunction(
    cache::_Cache,
    ci::MOI.ConstraintIndex{F, DisjunctionSet}
    ) where {F}
    set = MOI.get(cache, MOI.ConstraintSet(), ci)
    rows = _scalarize(MOI.get(cache, MOI.ConstraintFunction(), ci))
    disjunction_activation = _activation_affine(
        rows[activation_index(set)])
    disjuncts = _Disjunct[]
    for (i, j) in enumerate(indicator_indices(set))
        activation = _activation_affine(rows[j])
        binary, active_value = _activation_binary(activation)
        zero_one = MOI.ConstraintIndex{MOI.VariableIndex, MOI.ZeroOne}(
            binary.value)
        MOI.is_valid(cache, zero_one) || error("The indicator variable " *
            "of a `DisjunctionSet` disjunct must be `MOI.ZeroOne`.")
        # transcribed rows can carry function constants: shift them
        # into the scalar sets so consumers add constant-free rows
        normalized = [MOI.Utilities.normalize_constant(
            _as_row(_demote(rows[k])), inner) for (k, inner) in
            zip(row_indices(set, i), set.inner_sets[i])]
        push!(disjuncts, _Disjunct(activation, binary, active_value,
            MOI.AbstractScalarFunction[f for (f, _) in normalized],
            MOI.AbstractScalarSet[s for (_, s) in normalized]))
    end
    return _Disjunction(disjunction_activation, disjuncts)
end

# partition the cache for the LOA loop
function _build_problem(model::Optimizer)
    cache = model.cache
    sense = MOI.get(cache, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        # feasibility model: minimize a constant zero objective
        sense = MOI.MIN_SENSE
        objective = MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0)
    else
        F = MOI.get(cache, MOI.ObjectiveFunctionType())
        objective = _demote(MOI.get(cache, MOI.ObjectiveFunction{F}()))
    end
    variables = MOI.get(cache, MOI.ListOfVariableIndices())
    disjunctions = _Disjunction[]
    variable_cis = MOI.ConstraintIndex[]
    linear_cis = MOI.ConstraintIndex[]
    nonlinear_rows = Tuple{MOI.AbstractScalarFunction,
        MOI.AbstractScalarSet}[]
    for (FC, S) in MOI.get(cache, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(cache, MOI.ListOfConstraintIndices{FC, S}())
        if S === DisjunctionSet
            append!(disjunctions,
                _parse_disjunction(cache, ci) for ci in cis)
        elseif FC === MOI.VariableIndex && S <: MOI.AbstractScalarSet
            append!(variable_cis, cis)
        elseif FC === MOI.ScalarAffineFunction{Float64} &&
                S <: MOI.AbstractScalarSet
            append!(linear_cis, cis)
        elseif FC <: Union{MOI.ScalarQuadraticFunction{Float64},
                MOI.ScalarNonlinearFunction} && S <: MOI.AbstractScalarSet
            append!(nonlinear_rows,
                (_demote(MOI.get(cache, MOI.ConstraintFunction(), ci)),
                    MOI.get(cache, MOI.ConstraintSet(), ci))
                for ci in cis)
        else
            error("DisjunctiveAlgorithms does not support `$FC`-in-`$S` constraints.")
        end
    end
    binaries = unique!([disjunct.binary for disjunction in disjunctions
        for disjunct in disjunction.disjuncts])
    # non-indicator discrete variables keep their integrality; the
    # nlp_solver must handle whatever remains
    return _Problem(variables, binaries, disjunctions, variable_cis,
        linear_cis, nonlinear_rows, sense, objective)
end

function _disjunct_active(combination::AbstractDict, disjunct::_Disjunct)
    return combination[disjunct.binary] == disjunct.active_value
end

_is_nonlinear_disjunct(disjunct::_Disjunct) =
    any(!_is_linear(func) for func in disjunct.functions)

################################################################################
#                        INNER SOLVER SUPPORT CHECK
################################################################################
# Constraint types the master receives: variable constraints, linear
# rows, and the gated linear disjunct rows per `MasterReformulation`.
# The exactly-one rows, OA cuts, no-good cuts, and slack bounds are
# always affine rows or variable bounds.
function _master_constraint_types(model::Optimizer, problem::_Problem)
    affine = MOI.ScalarAffineFunction{Float64}
    required = Set{Tuple{Type, Type}}([
        (affine, MOI.EqualTo{Float64}),
        (affine, MOI.LessThan{Float64}),
        (affine, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.GreaterThan{Float64}),
        (MOI.VariableIndex, MOI.LessThan{Float64}),
    ])
    for ci in problem.variable_cis
        push!(required, (MOI.VariableIndex,
            typeof(MOI.get(model.cache, MOI.ConstraintSet(), ci))))
    end
    for ci in problem.linear_cis
        push!(required, (affine,
            typeof(MOI.get(model.cache, MOI.ConstraintSet(), ci))))
    end
    for (func, set) in problem.nonlinear_rows
        _is_linear(func) && push!(required, (affine, typeof(set)))
    end
    MOI.get(model, MasterReformulation()) == "indicator" || return required
    for disjunction in problem.disjunctions,
        disjunct in disjunction.disjuncts
        activate = disjunct.active_value ? MOI.ACTIVATE_ON_ONE :
            MOI.ACTIVATE_ON_ZERO
        for (func, set) in zip(disjunct.functions, disjunct.sets)
            _is_linear(func) || continue
            for inner in _indicator_sets(set)
                push!(required, (MOI.VectorAffineFunction{Float64},
                    MOI.Indicator{activate, typeof(inner)}))
            end
        end
    end
    return required
end

# Constraint types the NLP subproblems receive: every global and
# disjunct row plus the binary fixes and the NLPF slack variable.
function _nlp_constraint_types(model::Optimizer, problem::_Problem)
    required = Set{Tuple{Type, Type}}([
        (MOI.VariableIndex, MOI.EqualTo{Float64}),
        (MOI.VariableIndex, MOI.GreaterThan{Float64}),
    ])
    indicators = Set(problem.binaries)
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        vi in indicators && continue
        push!(required, (MOI.VariableIndex,
            typeof(MOI.get(model.cache, MOI.ConstraintSet(), ci))))
    end
    for ci in problem.linear_cis
        push!(required, (MOI.ScalarAffineFunction{Float64},
            typeof(MOI.get(model.cache, MOI.ConstraintSet(), ci))))
    end
    for (func, set) in problem.nonlinear_rows
        push!(required, (typeof(func), typeof(set)))
    end
    for disjunction in problem.disjunctions,
        disjunct in disjunction.disjuncts
        for (func, set) in zip(disjunct.functions, disjunct.sets)
            push!(required, (typeof(func), typeof(set)))
        end
    end
    return required
end

function _check_support(solver, name::String, destination::String, required)
    for (F, S) in required
        MOI.supports_constraint(solver, F, S) || error(
            "The `$name` ($(MOI.get(solver, MOI.SolverName()))) does " *
            "not support `$F`-in-`$S` constraints, which the " *
            "$destination requires.")
    end
    return
end

# Fail before any subproblem work when an inner solver cannot take the
# constraint types routed to it, naming the solver and the type. A
# subproblem method brings its own solver, so only the `nothing`
# (MOI-path) method checks the `nlp_solver`.
function _check_inner_support(method, model::Optimizer, problem::_Problem)
    mip = _instantiate(model.mip_solver)
    _check_support(mip, "mip_solver", "master problem",
        _master_constraint_types(model, problem))
    _check_nlp_support(method, model, problem)
    return
end

_check_nlp_support(method, ::Optimizer, ::_Problem) = nothing

function _check_nlp_support(::Nothing, model::Optimizer, problem::_Problem)
    nlp = _instantiate(model.nlp_solver)
    _check_support(nlp, "nlp_solver", "NLP subproblems",
        _nlp_constraint_types(model, problem))
    F = typeof(problem.objective)
    MOI.supports(nlp, MOI.ObjectiveFunction{F}()) || error(
        "The `nlp_solver` ($(MOI.get(nlp, MOI.SolverName()))) does " *
        "not support the `$F` objective the NLP subproblems require.")
    return
end
