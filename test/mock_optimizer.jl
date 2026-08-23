################################################################################
#                            MOCK SOLVER
################################################################################
# Delegates every MOI call to a wrapped optimizer, but sleeps
# `sleep_time` seconds in each solve and, from solve `fail_from` on,
# skips the inner solve and reports `fail_status` with no solution.
# Deterministic triggers for the deadline and abnormal-master paths.
# `strict_constants` imitates direct wrappers (e.g. Gurobi) that reject
# scalar functions with nonzero constants.
mutable struct MockSolver <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    sleep_time::Float64
    fail_from::Int
    fail_status::MOI.TerminationStatusCode
    strict_constants::Bool
    solves::Int
    failing::Bool
end

function MockSolver(
    factory;
    sleep_time::Float64 = 0.0,
    fail_from::Int = typemax(Int),
    fail_status::MOI.TerminationStatusCode = MOI.NODE_LIMIT,
    strict_constants::Bool = false
    )
    return MockSolver(MOI.instantiate(factory), sleep_time, fail_from,
        fail_status, strict_constants, 0, false)
end

function MOI.optimize!(model::MockSolver)
    model.solves += 1
    model.sleep_time > 0 && sleep(model.sleep_time)
    model.failing = model.solves >= model.fail_from
    model.failing || MOI.optimize!(model.inner)
    return
end

const _WrappedAttr = Union{MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute}
const _WrappedIndexAttr = Union{MOI.AbstractVariableAttribute,
    MOI.AbstractConstraintAttribute}
const _WrappedIndex = Union{MOI.VariableIndex, MOI.ConstraintIndex}

function MOI.get(model::MockSolver, attr::_WrappedAttr)
    if model.failing
        attr isa MOI.TerminationStatus && return model.fail_status
        attr isa MOI.PrimalStatus && return MOI.NO_SOLUTION
    end
    return MOI.get(model.inner, attr)
end

MOI.is_empty(model::MockSolver) = MOI.is_empty(model.inner)
MOI.empty!(model::MockSolver) = MOI.empty!(model.inner)
MOI.supports_incremental_interface(::MockSolver) = true
MOI.copy_to(model::MockSolver, src::MOI.ModelLike) =
    MOI.copy_to(model.inner, src)
MOI.add_variable(model::MockSolver) = MOI.add_variable(model.inner)
MOI.delete(model::MockSolver, index::_WrappedIndex) =
    MOI.delete(model.inner, index)
MOI.is_valid(model::MockSolver, index::_WrappedIndex) =
    MOI.is_valid(model.inner, index)

function MOI.add_constraint(
    model::MockSolver,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet
    )
    if model.strict_constants && func isa Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64}} && !iszero(func.constant)
        throw(MOI.ScalarFunctionConstantNotZero{
            Float64, typeof(func), typeof(set)}(func.constant))
    end
    return MOI.add_constraint(model.inner, func, set)
end

function MOI.supports_constraint(
    model::MockSolver,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet}
    )
    return MOI.supports_constraint(model.inner, F, S)
end

MOI.supports(model::MockSolver, attr::_WrappedAttr) =
    MOI.supports(model.inner, attr)

MOI.set(model::MockSolver, attr::_WrappedAttr, value) =
    MOI.set(model.inner, attr, value)

function MOI.supports(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    I::Type{<:_WrappedIndex}
    )
    return MOI.supports(model.inner, attr, I)
end

function MOI.get(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    index::_WrappedIndex
    )
    return MOI.get(model.inner, attr, index)
end

function MOI.set(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    index::_WrappedIndex,
    value
    )
    return MOI.set(model.inner, attr, index, value)
end
