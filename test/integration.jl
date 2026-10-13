using DisjunctiveProgramming, HiGHS, Ipopt, InfiniteOpt

function _optimizer_factory()
    return () -> DA.Optimizer(Ipopt.Optimizer, HiGHS.Optimizer)
end

# The same GDP solved through a BigM reformulation and through the
# lowering into DisjunctiveAlgorithms must agree: max x with
# (x <= 3) or (x <= 7).
function test_lowering_solve_linear()
    model = GDPModel(_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = Direct())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-4
    @test value(x) ≈ 7.0 atol = 1e-4
    @test value(Y[2])

    reference = GDPModel(HiGHS.Optimizer)
    set_silent(reference)
    @variable(reference, 0 <= x2 <= 10)
    @variable(reference, Y2[1:2], Logical)
    @constraint(reference, x2 <= 3, Disjunct(Y2[1]))
    @constraint(reference, x2 <= 7, Disjunct(Y2[2]))
    @disjunction(reference, Y2)
    @objective(reference, Max, x2)
    optimize!(reference, gdp_method = BigM())
    @test objective_value(model) ≈ objective_value(reference) atol = 1e-6
end

# Nonlinear disjunct row: max x with (x <= 3) or (x^2 == 64); the
# unique optimum 8 is checked directly
function test_lowering_solve_nonlinear()
    model = GDPModel(_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 == 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = Direct())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2])
end

# Nested solve: max x with x <= 2, or a nested mode choice between
# x^2 <= 25 and x <= 8.
function test_lowering_solve_nested()
    model = GDPModel(_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 2, Disjunct(Y[1]))
    @constraint(model, x^2 <= 25, Disjunct(W[1]))
    @constraint(model, x <= 8, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = Direct())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2])
    @test value(W[2])
end

# An InfiniteGDPModel lowers through the same method: the disjunction
# transcribes to one DisjunctionSet constraint per support.
function test_lowering_infinite()
    model = InfiniteGDPModel(_optimizer_factory())
    set_silent(model)
    @infinite_parameter(model, t in [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, integral(x, t))
    optimize!(model, gdp_method = Direct())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 10.0 atol = 1e-4
    @test all(value(x) .>= 5.0 .- 1e-4)
end

# A nested infinite disjunction transcribes with the parent indicator
# as its activation, so the rebuilt constraint function is a uniform
# vector of scalar expressions (no constant to promote).
function test_lowering_infinite_nested()
    model = InfiniteGDPModel(_optimizer_factory())
    set_silent(model)
    @infinite_parameter(model, t in [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @variable(model, W[1:2], InfiniteLogical(t))
    @constraint(model, x <= 2, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(W[1]))
    @constraint(model, x >= 3, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, integral(x, t))
    optimize!(model, gdp_method = Direct())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 10.0 atol = 1e-4
end

@testset "DisjunctiveProgramming integration" begin
    test_lowering_solve_linear()
    test_lowering_solve_nonlinear()
    test_lowering_solve_nested()
    test_lowering_infinite()
    test_lowering_infinite_nested()
end
