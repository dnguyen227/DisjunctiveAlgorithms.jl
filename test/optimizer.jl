# A two-disjunct toy stored straight into a model: binaries z1, z2 and
# continuous x with x <= 0 (disjunct 1) or x >= 1 (disjunct 2).
function _build_toy_disjunction(model)
    x = MOI.add_variable(model)
    z = MOI.add_variables(model, 2)
    for zi in z
        MOI.add_constraint(model, zi, MOI.ZeroOne())
    end
    set = DA.DisjunctionSet([[MOI.LessThan(0.0)], [MOI.GreaterThan(1.0)]])
    func = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, z[1])),
         MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, z[2])),
         MOI.VectorAffineTerm(4, MOI.ScalarAffineTerm(1.0, x)),
         MOI.VectorAffineTerm(5, MOI.ScalarAffineTerm(1.0, x))],
        [1.0, 0.0, 0.0, 0.0, 0.0]) # constant activation in component 1
    ci = MOI.add_constraint(model, func, set)
    return x, z, set, ci
end

function test_optimizer_scaffold()
    optimizer = DA.Optimizer(nothing)
    @test MOI.is_empty(optimizer)
    @test MOI.get(optimizer, MOI.SolverName()) == "DisjunctiveAlgorithms"
    @test MOI.get(optimizer, MOI.TerminationStatus()) ==
        MOI.OPTIMIZE_NOT_CALLED
    @test MOI.get(optimizer, MOI.ResultCount()) == 0
    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
end

function test_optimizer_options()
    optimizer = DA.Optimizer(nothing)
    @test MOI.get(optimizer, DA.Algorithm()) === nothing
    @test MOI.supports(optimizer, DA.Algorithm())
    @test MOI.supports(optimizer, DA.NumIterationLimit())
    @test MOI.get(optimizer, DA.NumIterationLimit()) == 10
    # setting an attribute materializes the default algorithm
    MOI.set(optimizer, DA.NumIterationLimit(), 5)
    @test MOI.get(optimizer, DA.NumIterationLimit()) == 5
    @test MOI.get(optimizer, DA.Algorithm()) isa DA.LOA
    @test MOI.get(optimizer, DA.M_Value()) == 1e9
    algorithm = DA.LOA()
    MOI.set(optimizer, DA.Algorithm(), algorithm)
    @test MOI.get(optimizer, DA.Algorithm()) === algorithm
    @test MOI.get(optimizer, DA.NumIterationLimit()) == 10
    @test !MOI.supports(optimizer, MOI.RawOptimizerAttribute("max_iter"))
    MOI.set(optimizer, MOI.Silent(), true)
    @test MOI.get(optimizer, MOI.Silent())
    @test MOI.get(optimizer, MOI.TimeLimitSec()) == 3600.0
    MOI.set(optimizer, MOI.TimeLimitSec(), 100)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) == 100.0
    MOI.set(optimizer, MOI.TimeLimitSec(), nothing)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
end

function test_model_building()
    optimizer = DA.Optimizer(nothing)
    x, z, set, ci = _build_toy_disjunction(optimizer)
    @test MOI.is_valid(optimizer, ci)
    @test MOI.get(optimizer, MOI.NumberOfVariables()) == 3
    @test (MOI.VectorAffineFunction{Float64}, DA.DisjunctionSet) in
        MOI.get(optimizer, MOI.ListOfConstraintTypesPresent())
    @test MOI.get(optimizer, MOI.ConstraintSet(), ci) == set
    objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        objective)
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    @test !MOI.is_empty(optimizer)
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)
end

function test_copy_to()
    src = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x, z, set, ci = _build_toy_disjunction(src)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0))
    dest = DA.Optimizer(nothing)
    index_map = MOI.copy_to(dest, src)
    @test MOI.get(dest, MOI.NumberOfVariables()) == 3
    @test (MOI.VectorAffineFunction{Float64}, DA.DisjunctionSet) in
        MOI.get(dest, MOI.ListOfConstraintTypesPresent())
    @test MOI.get(dest, MOI.ConstraintSet(), index_map[ci]) == set
    dest_func = MOI.get(dest, MOI.ConstraintFunction(), index_map[ci])
    @test dest_func.terms[1].scalar_term.variable == index_map[z[1]]
end

# The incremental MOI surface forwards to the cache: names, starts,
# constrained variables, and deletion.
function test_moi_forwarding()
    optimizer = DA.Optimizer(nothing)
    @test MOI.supports_incremental_interface(optimizer)
    x = MOI.add_variable(optimizer)
    MOI.set(optimizer, MOI.VariableName(), x, "x")
    @test MOI.get(optimizer, MOI.VariableIndex, "x") == x
    MOI.set(optimizer, MOI.VariablePrimalStart(), x, 2.5)
    @test MOI.get(optimizer, MOI.VariablePrimalStart(), x) == 2.5
    y, ci_y = MOI.add_constrained_variables(optimizer, MOI.Nonnegatives(2))
    @test length(y) == 2
    @test MOI.is_valid(optimizer, ci_y)
    func = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    ci = MOI.add_constraint(optimizer, func, MOI.LessThan(1.0))
    MOI.set(optimizer, MOI.ConstraintName(), ci, "c")
    @test MOI.get(optimizer, typeof(ci), "c") == ci
    MOI.delete(optimizer, ci)
    @test !MOI.is_valid(optimizer, ci)
    @test !MOI.supports_add_constrained_variables(optimizer,
        DA.DisjunctionSet)
end

# A constraint type the partition cannot place errors at solve time.
function test_unsupported_constraint_type()
    optimizer = DA.Optimizer(nothing)
    x = MOI.add_variable(optimizer)
    func = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x))], [0.0])
    MOI.add_constraint(optimizer, func, MOI.Nonnegatives(1))
    @test_throws ErrorException MOI.optimize!(optimizer)
end

# Only the scalar objective functions the solve demotes are supported;
# a vector objective is refused at set time. The solver version comes
# from Project.toml, not a copy.
function test_objective_function_support()
    optimizer = DA.Optimizer(nothing)
    @test MOI.supports(optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test MOI.supports(optimizer,
        MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}())
    @test !MOI.supports(optimizer,
        MOI.ObjectiveFunction{MOI.VectorAffineFunction{Float64}}())
    x = MOI.add_variable(optimizer)
    func = MOI.Utilities.operate(vcat, Float64, 1.0 * x, 2.0 * x)
    attr = MOI.ObjectiveFunction{typeof(func)}()
    @test_throws MOI.UnsupportedAttribute MOI.set(optimizer, attr, func)
    @test MOI.get(optimizer, MOI.SolverVersion()) ==
        string(pkgversion(DA))
end

@testset "Optimizer scaffold" begin
    test_optimizer_scaffold()
    test_optimizer_options()
    test_model_building()
    test_copy_to()
    test_moi_forwarding()
    test_unsupported_constraint_type()
    test_objective_function_support()
end
