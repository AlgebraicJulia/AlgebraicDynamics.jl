using AlgebraicDynamics
using AlgebraicDynamics.Systems
using Test

using Catlab
using Catlab.Theories
using Catlab.WiringDiagrams
using Catlab.Programs
using RecursiveArrayTools
using OrdinaryDiffEq

include("discdynam.jl")
include("hypergraphs.jl")
include("linrels.jl")

R = Ob(FreeSMC, Float64)
id(R)
sinesys = System(Int[], [1.0], [1], x -> -x)
@testset "AlgebraicDynamics.jl" begin
    # Write your own tests here.


@testset "ArrayPartition" begin
    v = ArrayPartition([1,2,3],[4,5,6])
    @test v[1] == 1
    @test v[2] == 2
    @test v[5] == 5
    @test v.x[1] == [1,2,3]
    @test v.x[2] == [4,5,6]

    v = ArrayPartition([1,2,3], ArrayPartition([4,5],[6]), [7,8,9])
    @test v[5] == 5
    @test v[6] == 6
end

@testset "dom_ids" begin
    f = Hom(System([1], [1.0], [1], (x, t)->-x), R, R)
    @test dom_ids(compose(f,f)) == [1]
    @test codom_ids(compose(f,f)) == [2]

    @test dom_ids(otimes(f,f)) == [1, 2]
    @test codom_ids(otimes(f,f)) == [1, 2]

    @test dom_ids(otimes(compose(f,f),f)) == [1, 3]
    @test codom_ids(otimes(compose(f,f),f)) == [2, 3]
end

@testset "Initforward" begin
    @test initforward(id(R), 1.0) == zeros(Float64, 1)
    @test initforward(id(otimes(R, R)), 1.0) == zeros(Float64, 2)
    @test initforward(id(otimes(R, otimes(R, R))), 1.0) == zeros(Float64, 3)

    f = Hom(System([1], [1.0], [1], (x, t)->-x), R, R)
    @test initforward(f, 1) == [-1]
    @test initforward(compose(f,f), 1) == [-2, -2]
    @test initforward(otimes(f,f), 1) == [-1,-1]
    @test initforward(otimes(compose(f,f),f), 1) == [-2, -2,-1]
end

@testset "SIR" begin
    si = Hom(System([1], [99, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    sirtest = Hom(System([1], [99,1,0], [4], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]+ -0.05x[2], 0.05x[2]]), R, R)

    @test collect(initforward(sir, 1))[[1,2,4]] == initforward(sirtest, 1)

    @test states(si) == [99, 1]
    @test states(ir) == [1, 0]
    @test states(sir) == [99, 1, 1, 0]
    @test states(sir)[1] == 99
    @test states(sir)[2] == 1
    @test states(sir)[3] == 1

    @test length(forward(sir, states(sir), 0)) == 4
end
@testset "ODESolvers" begin

    si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    p = problem(sir, (0,270.0))
    sol = OrdinaryDiffEq.solve(p, alg=Tsit5())
    @test sol.u[end][1] < 1e-1
    @test sol.u[end][end] > 100-1

    si = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0035x[1]*x[2], 0.0035x[1]*x[2]]), R,R)
    ir = Hom(System([1], [1.00, 0], [2], (x,t)->[-0.05x[1], 0.05x[1]]), R, R)
    sir = compose(si, ir)
    si_2 = Hom(System([1], [99.0, 1], [2], (x,t)->[-0.0025x[1]*x[2], 0.0025x[1]*x[2]]), R,R)
    ir_2= Hom(System([1], [1.00, 0], [2], (x,t)->[-0.07x[1], 0.07x[1]]), R, R)
    sir_2 = compose(si_2, ir_2)
    p = problem(otimes(sir, sir_2), (0,270.0))
    sol = OrdinaryDiffEq.solve(p, alg=Tsit5())
    @test sol.u[end][1] < 1e-1
    @test sol.u[end][4] > 100-1
    @test sol.u[end][5] < 5
    @test sol.u[end][end] > 100-5
end
end
