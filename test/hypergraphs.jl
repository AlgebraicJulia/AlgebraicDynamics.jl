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

    function vectorfield!(d, generators::Dict, scratch::AbstractVector)
        # function dynam(d, generators::Dict)
        #     # we create a task array so that you could parallelize the computation of
        #     # primitive subsystems using pmap. This function overhead could be eliminated
        #     # for benchmarking the sequential version.
        #     tasks = Function[]
        #     for box in 1:nparts(d, :Box)
        #         n = subpart(d, :name)[box]
        #         ports = incident(d, box, :box)
        #         juncs = [subpart(d,:junction)[p] for p in ports]
        #         tk = (du, u, θ, t) -> generators[n](u[juncs], θ, t)
        #         push!(tasks, tk)
        #     end
        #     # this function could avoid doing all the lookups every time by enclosing the ports
        #     # vectors into the function.
        #     # TODO: Eliminate all allocations here
        #     aggregate!(out, du) = for j in 1:nparts(d, :Junction)
        #         ports = incident(d, j, :junction)
        #         out[j] = sum(du[ports])
        #     end
        #     return tasks, aggregate!
        # end
        # tasks, aggregate! = dynam!(d,generators)
        # @show juncs = subpart(d, :junction)
        # @show boxes = subpart(d, :box)
        task(du, u, p, t) = begin
            map(1:nparts(d, :Box)) do b
                boxports = incident(d, b, :box)
                juncs = subpart(d, :junction)
                boxjuncs = juncs[boxports]
                tv = view(scratch, boxports)
                v = view( u, boxjuncs)
                n = subpart(d,:name)[b]
                generators[n](tv, v, p[b], t)
            end
            map(1:nparts(d, :Junction)) do j
                juncports = incident(d, j, :junction)
                du[j] = sum(scratch[juncports])
            end
            return du
        end
        return task
    end

    @testset "In Place Version" begin
        g = Dict(
            :birth     => (du, u, p, t) -> begin du[1] =  α*u[1] end,
            :death     => (du, u, p, t) -> begin du[1] = -γ*u[1] end,
            :predation => (du, u, p, t) -> begin du[1] = -β*u[1]*u[2]
            du[2] = δ*u[1]*u[2]
            end
        )
        scratch = zeros(nparts(d, :Port))
        println("In Place Creation")
        @time f = vectorfield!(d, g, scratch)
        du = zeros(nparts(d, :Junction))
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
        @time sol = OrdinaryDiffEq.solve(p, Tsit5())
        @time sol = OrdinaryDiffEq.solve(p, Tsit5())
        @test all(vcat(sol.u...) .> 0)
    end
end
