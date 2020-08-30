using AlgebraicDynamics
using AlgebraicDynamics.Hypergraphs

using Base.Iterators
using Test
using LinearAlgebra
using Catlab
using Catlab.Programs.RelationalPrograms
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.CSets

using OrdinaryDiffEq

@testset "Hypergraphs" begin
    α = 1.2
    β = 0.1
    γ = 1.3
    δ = 0.1

    g = Dict(
        :birth     => (u, p, t) -> [ α*u[1]],
        :death     => (u, p, t) -> [-γ*u[1]],
        :predation => (u, p, t) -> [-β*u[1]*u[2], δ*u[1]*u[2]],
    )

    d = @relation (x,y) where (x::X, y::X) begin
        birth(x)
        predation(x,y)
        death(y)
    end

    tasks, aggregate! = dynam(d, g)
    @test length(tasks) == 3

    u₀ = [13.0, 12.0]

    nullparams = collect(repeat([0], length(tasks)))
    @test norm(vectorfield(d, g)(zeros(nparts(d,:Junction)), u₀, nullparams, 0)) <= 1e-6

    @time f = vectorfield(d,g)
    p = ODEProblem(f, u₀, (0,10.0), nullparams)

    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test norm(sol.u[end] - u₀) < 1e-4

    u₀ = [17.0, 11.0]
    p = ODEProblem(f, u₀, (0,10.0), nullparams)
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test all(vcat(sol.u...) .> 0)

end
