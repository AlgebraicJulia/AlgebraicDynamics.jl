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
@testset "Out of Place LV" begin
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
    @time f = vectorfield(d,g)
    @test norm(f(zeros(nparts(d,:Junction)), u₀, nullparams, 0)) <= 1e-12

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


@testset "In Place Version" begin
    α = 1.2
    β = 0.1
    γ = 1.3
    δ = 0.1

    d = @relation (x,y) where (x::X, y::X) begin
        birth(x)
        predation(x,y)
        death(y)
    end
    g = Dict(
        :birth     => (du, u, p, t) -> begin du[1] =  α*u[1] end,
        :death     => (du, u, p, t) -> begin du[1] = -γ*u[1] end,
        :predation => (du, u, p, t) -> begin du[1] = -β*u[1]*u[2]
        du[2] = δ*u[1]*u[2]
        end
    )
    scratch = zeros(nparts(d, :Port))
    nullparams = zeros(nparts(d, :Box))
    du = zeros(nparts(d, :Junction))
    println("In Place Creation")
    @time f = vectorfield!(d, g, scratch)
    f(du, [13.0, 12.0], zeros(3), 0.0)
    @test norm(du) < 1e-12

    u₀ = [13.0, 12.0]
    p = ODEProblem(f, u₀, (0,10.0), nullparams)
    println("In Place Solve EQ")
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test norm(sol.u[end] - u₀) < 1e-4

    u₀ = [17.0, 11.0]
    p = ODEProblem(f, u₀, (0,10.0), nullparams)
    println("In Place Solve Osc")
    f(du, [17.0, 11.0], zeros(3), 0.0)
    @test norm(du) > 1e-4
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test all(vcat(sol.u...) .> 0)
    @test norm(u[1]-u₀[1] for (u,t) in tuples(sol)) > 1e-2
end

@testset "Food Web" begin
    α = 1.2
    β = 0.1
    γ = 1.3
    δ = 0.1
    g = Dict(
        :birth     => (u, p, t) -> [ α*u[1] ],
        :death     => (u, p, t) -> [-γ*u[1] ],
        :predation => (u, p, t) -> [-β*u[1]*u[2], δ*u[1]*u[2]]
    )
    d = @relation (x,y,z) where (x::X, y::X, z::X) begin
        birth(x)
        predation(x,y)
        death(y)
        predation(x,z)
        death(z)
    end
    @show d
    nullparams = zeros(nparts(d, :Box))
    du         = zeros(nparts(d, :Junction))
    println("Out of Place Creation")
    @time f = vectorfield(d, g)
    f(du, [13.0, 12.0, 12], nullparams, 0.0)
    @show du
    f(du, [13.0, 12.0, 6], nullparams, 0.0)
    @show du
    f(du, [17.0, 12.0, 6], nullparams, 0.0)
    @show du

    u₀ = [13.0, 12.0, 12]
    p = ODEProblem(f, u₀, (0,10.0), nullparams)
    println("Out of Place Solve EQ")
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test norm(u[2]-u[3] for (u,t) in tuples(sol)) < 1e-2

    u₀ = [17.0, 11.0, 6]
    p = ODEProblem(f, u₀, (0,10.0), nullparams)
    println("Out of Place Solve Osc")
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test all(vcat(sol.u...) .> 0)
    @test norm(u[2]-u[3] for (u,t) in tuples(sol)) > 1e-2

    # du         = zeros(nparts(d, :Junction))
    # scratch    = zeros(nparts(d, :Port))
    # g = Dict(
    #     :birth     => (du, u, p, t) -> begin du[1] =  α*u[1] end,
    #     :death     => (du, u, p, t) -> begin du[1] = -γ*u[1] end,
    #     :predation => (du, u, p, t) -> begin du[1] = -β*u[1]*u[2]
    #     du[2] = δ*u[1]*u[2]
    #     end
    # )
    # println("In Place Creation")
    # @time f = vectorfield!(d, g, scratch)
    # f(du, [13.0, 12.0, 12], nullparams, 0.0)
    # @show du
    # f(du, [13.0, 12.0, 6], nullparams, 0.0)
    # @show du
    # f(du, [17.0, 12.0, 6], nullparams, 0.0)
    # @show du

    # u₀ = [13.0, 12.0, 12]
    # p = ODEProblem(f, u₀, (0,10.0), nullparams)
    # println("In Place Solve EQ")
    # @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    # @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    # @test norm(sol.u[end] - u₀) < 1e-4
    # @show sol.u

    # u₀ = [17.0, 11.0, 6]
    # p = ODEProblem(f, u₀, (0,10.0), nullparams)
    # println("In Place Solve Osc")
    # @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    # @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    # @test all(vcat(sol.u...) .> 0)
    # @show sol.u
end
    @testset "Food Web Parameters" begin
    g = Dict(
        :birth     => (u, p, t) -> [ p[1]*u[1] ],
        :death     => (u, p, t) -> [-p[1]*u[1] ],
        :predation => (u, p, t) -> [-p[1]*u[1]*u[2], p[2]*u[1]*u[2]]
    )
    d = @relation (x,y,z) where (x::X, y::X, z::X) begin
        birth(x)
        predation(x,y)
        death(y)
        predation(x,z)
        death(z)
    end
    @show d
    params = [[1.2], [0.1, 0.1], [1.3], [0.05, 0.05], [1.1]]
    du         = zeros(nparts(d, :Junction))
    println("Out of Place Creation")
    @time f = vectorfield(d, g)
    f(du, [13.0, 12.0, 12], params, 0.0)
    @show du
    f(du, [13.0, 12.0, 6], params, 0.0)
    @show du
    f(du, [17.0, 12.0, 6], params, 0.0)
    @show du

    u₀ = [13.0, 12.0, 12]
    p = ODEProblem(f, u₀, (0,10.0), params)
    println("Out of Place Solve EQ")
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test norm(u[2]-u[3] for (u,t) in tuples(sol)) > 1e-2

    u₀ = [17.0, 11.0, 6]
    p = ODEProblem(f, u₀, (0,10.0), params)
    println("Out of Place Solve Osc")
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @time sol = OrdinaryDiffEq.solve(p, Tsit5())
    @test all(vcat(sol.u...) .> 0)
    @test norm(u[2]-u[3] for (u,t) in tuples(sol)) > 1e-2
    end

end
