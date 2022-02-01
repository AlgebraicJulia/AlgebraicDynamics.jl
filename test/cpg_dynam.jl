using AlgebraicDynamics.CPortGraphDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam: draw, barbell, gridpath, grid, meshpath
using AlgebraicDynamics.DWDDynam: trajectory
using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs
using Catlab.CategoricalAlgebra
using Catlab.Theories
using Catlab.Graphs
using Catlab

using DelayDiffEq
using LabelledArrays

using Test
using PrettyTables
using Base.Iterators: repeated


function simulate(f::ContinuousMachine{T}, nsteps::Int, h::Real, u0::Vector, xs=T[]) where T
    trajectory(euler_approx(f, h), u0, xs, nothing, nsteps)
end

function printsim(traj, stepfun, indxfun, shape)
    for u in stepfun(traj)
        pretty_table(reshape(indxfun(u), shape), equal_columns_width=true, noheader=true)
    end
end

d = barbell(2)
xs = [
    ContinuousMachine{Float64}(2,2,(u, x, p, t)->[x[1]*u[1], x[2]*u[2]], (u,p,t)->u),
    ContinuousMachine{Float64}(2,2,(u, x, p, t)->[1/x[1]*u[1], -x[2]*u[2]], (u,p,t)->u)
]
h = 0.1
u₀ = ones(Float64, 4)
composite = oapply(d, xs)
fcomp(u,p,t) = eval_dynamics(composite, u, Float64[], p, t)
rcomp(u) = readout(composite,u)
@test rcomp(u₀) == []
u₁ = fcomp(u₀, Float64[], h)
@test u₁ == [1,1, 1, -1]
@test rcomp(u₁) == []
@test fcomp(u₁, [], h) == [1,-1, 1, 1]
# @show simulate(composite, 10, h, u₀)

d₀ = OpenCPortGraph()
add_parts!(d₀, :Box, 1)
d₁ = barbell(2)
F = ACSetTransformation((Box=[2],), d₀, d₁)
G = ACSetTransformation((Box=[1],), d₀, d₁)
# |1| <-> |3| <-> |5|
# |2| <-> |4| <-> |6|
d₂ = apex(pushout(F,G))
Catlab.Theories.id(OpenCPortGraph, n) = begin
    g = OpenCPortGraph()
    add_parts!(g, :Box, 1)
    add_parts!(g, :Port, n, box=1)
    add_parts!(g, :OuterPort, n, con=1:n)
    return g
end
lob(n) = let
    b = barbell(n)
    p = add_parts!(b, :Port, n, box=1)
    add_parts!(b, :OuterPort, n, con=p)
    b
end
d₂′ = ocompose(barbell(2), [id(OpenCPortGraph, 2), lob(2)])
# @show d₂′
β = 0.4
μ = 0.4
α₁ = 0.01
α₂ = 0.01

sirfuncb = (u,x,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-x[1]), # Ṡ
                        β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-x[2]), #İ
                        μ*u[2] # Ṙ
                        ]
sirfuncm = (u,x,p,t)->[-β*u[1]*u[2] - α₁*(u[1]-(x[1]+x[3])/2),
                        β*u[1]*u[2] - μ*u[2] - α₂*(u[2]-(x[2]+x[4])/2),
                        μ*u[2]
                        ]

boundary  = ContinuousMachine{Float64}(2,3,sirfuncb, (u,p,t)->u[1:2])
middle    = ContinuousMachine{Float64}(4,3, sirfuncm, (u,p,t)->u[[1,2,1,2]])
threecity = oapply(d₂, [boundary,middle,boundary])

# println("Simulating 3 city")
traj = simulate(threecity, 100, 0.01, [100,1,0,100,0,0,100,0,0.0] )
# map(traj) do u
#     return (i1=u[2], i2=u[5], i3=u[8])
# end |> pretty_table

gl = @acset OpenCPortGraph begin
    Box = 3
    Port = 7
    Wire = 4
    OuterPort = 3
    box = [1,1,2,2,2,3,3]
    src = [2, 3, 5, 6]
    tgt = [3, 2, 6, 5]
    con = [1,4,7]
end

gm = @acset OpenCPortGraph begin
    Box = 3
    Port = 10
    Wire = 4
    OuterPort = 6
    box = [1,1,1,2,2,2,2,3,3,3]
    src = [2, 4, 6, 8]
    tgt = [4, 2, 8, 6]
    con = [3, 7, 10, 1, 5, 9]
end

gr = @acset OpenCPortGraph begin
    Box = 3
    Port = 7
    Wire = 4
    OuterPort = 3
    box = [1,1,2,2,2,3,3]
    src = [1, 3, 4, 6]
    tgt = [3, 1, 6, 4]
    con = [2,5,7]
end

symedges(g) = g.tables.W[g.tables.W.src .<= g.tables.W.tgt]
sympairs(z) = Iterators.filter(x->x[1] <= x[2], z)
pg2 = ocompose(barbell(3), [gl, gm])
g2 = migrate!(Graph(), pg2)
@test g2[ 9:11, :src] == [1,2,3]
@test g2[ 9:11, :tgt] == [4,5,6]
@test g2[12:14, :src] == [4,5,6]
@test g2[12:14, :tgt] == [1,2,3]

d3 = ocompose(barbell(3), [id(OpenCPortGraph, 3), lob(3)])
pg3 = ocompose(d3, [gl,gm,gr])
g3 = migrate!(Graph(), pg3)
@test g3[19:24, :src] == 1:6
@test g3[19:24, :tgt] == [4,5,6,1,2,3]
@test incident(pg3, 5, :box) == 11:14

α₁ = 1
fm = ContinuousMachine{Float64}(4, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 4)))
fl = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fr = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
ft = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fb = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fc = ContinuousMachine{Float64}(2, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 2)))

@test eval_dynamics(ft, ones(1), ones(3), nothing, 0.0) == zeros(1)
f₁ = oapply(gl, [fc, fl, fc])
f₂ = oapply(gm, [ft, fm, fb])
f₃ = oapply(gr, [fc, fr, fc])

F = oapply(pg3, [fc, fl, fc, ft, fm, fb, fc, fr, fc])
@test eval_dynamics(F, ones(Float64, 9), [], nothing, 1.0) == zeros(Float64, 9)
u₀ = zeros(Float64, 9)
u₀[5] = 1.0
@test eval_dynamics(F, u₀, [], nothing, 1.0)[5] < 0
@test eval_dynamics(F, u₀, [], nothing, 1.0)[4] == α₁
@test eval_dynamics(F, u₀, [], nothing, 1.0)[2] == α₁
@test eval_dynamics(F, u₀, [], nothing, 1.0)[3] == 0

d4 = @acset OpenCPortGraph begin
    Box = 3
    Port = 18
    Wire = 12
    OuterPort = 6
    box = Iterators.flatten([repeated(i, 6) for i in 1:3])
    con = [1,2,3,16,17,18]
    src = [4,5,6,7,8,9,10,11,12,13,14,15]
    tgt = [7,8,9,4,5,6,13,14,15,10,11,12]
end
pg3 = ocompose(d4, [gm,gm,gm])
draw(pg3)
# pg3[:, :con]

@testset "Laplacians" begin
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), ones(3), ones(6), nothing, 1.0) == zeros(3)
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), [1,2,1], ones(6), nothing, 1.0) == [1,-4,1]
    @test eval_dynamics(oapply(gm, [ft, fm, fb]), [1,2,0], ones(6), nothing, 1.0) == [1,-5, 4]

    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    @test nstates(F) == 9
    @test ninputs(F) == 6
    @test eval_dynamics(F, ones(9), ones(6), nothing, 1.0) == zeros(9)
    @test eval_dynamics(F, 2*ones(9), 2*ones(6), nothing, 1.0) == zeros(9)

    pg4 = ocompose(d4, [pg3, pg3, pg3])
    pg5 = ocompose(d4, [pg4, pg4, pg4])
    # draw(pg5)

    @test (nstates(F),ninputs(F)) == (9,6)
    F2 = oapply(d4, [F, F, F])
    @test (nstates(F2),ninputs(F2)) == (27,6)
    @test eval_dynamics(F2, ones(Float64, 27), ones(Float64, 6), nothing, 0.0) == zeros(27)
    F3 = oapply(d4, [F2,F2,F2])
    @test (nstates(F3), ninputs(F3)) == (81, 6)
    @test eval_dynamics(F3, ones(Float64, 81), ones(Float64, 6), nothing, 0.0) == zeros(81)
end

@testset "Advection-Diffusion" begin
    advecdiffuse(α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        fm = ContinuousMachine{Float64}(4, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 4)))
        fb = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        return ft, fm, fb
    end

    ft, fm, fb = advecdiffuse(1.0,2.0)
    eval_dynamics(ft, [1.0], [1,1,1.0], 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    eval_dynamics(F, zeros(9), ones(6), 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = advecdiffuse(1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] == traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] == traj[i][end-2:end][3]

    traj = simulate(F2, 16, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3]
end

@testset "Reaction-Diffusion-Advection" begin
    RDA(α₀, α₁, α₂) = begin
        diffop(u,p,t) = α₁ .* (sum(p) .- u .* length(p))
        advop(u,p,t)  = α₂ .* (p[end] .- u)
        ft = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        fm = ContinuousMachine{Float64}(4, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 4)))
        fb = ContinuousMachine{Float64}(3, 1, (u,p,q,t) -> α₀ .* u .+ diffop(u,p,u) .+ advop(u,p,t), (u,p,t)->collect(repeated(u[1], 3)))
        return ft, fm, fb
    end

    ft, fm, fb = RDA(0.1, 1.0,2.0)
    eval_dynamics(ft, [1.0], [1,1,1.0], nothing, 0.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    eval_dynamics(F, zeros(9), ones(6), nothing, 0.0)
    traj = simulate(F, 48, 0.1, zeros(9), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,3))

    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))

    ft, fm, fb = RDA(0.1, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 16, 0.1, zeros(27), vcat(ones(3), zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    i=16
    @test traj[i][end-2:end][1] ≈ traj[i][end-2:end][2] atol=1e-3
    @test traj[i][end-2:end][2] ≈ traj[i][end-2:end][3] atol=1e-3

    traj = simulate(F2, 64, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
    @test traj[i][end-2:end][1] <= traj[i][end-2:end][2]
    @test traj[i][end-2:end][2] >= traj[i][end-2:end][3]

    ft, fm, fb = RDA(-0.4, 1.0,4.0)
    F = oapply(d4, oapply(gm, [ft,fm,fb]))
    F2 = oapply(d4, [F,F,F])
    traj = simulate(F2, 128, 0.1, zeros(27), vcat([0,1,0], zeros(3)))
    # printsim(traj, t->t[end-2:end], identity, (3,9))
end

@testset "Grids" begin
    @testset for (n, m) in Iterators.product(1:6, 2:4)
        g = grid(n, m)
        @test nparts(g, :Box) == n*m
        @test nparts(g, :Port) == 6n + 4n*(m-2)
        @test nparts(g, :Wire) == 2*((n-1)*m + (m-1)*n)
    end
end

@testset "Ross-Macdonald model" begin
    c = @acset OpenCPortGraph begin
        Box = 2
        Port = 2
        Wire = 2
        OuterPort = 0
        box = [1,2]
        src = [1,2]
        tgt = [2,1]
        con = []
    end
    dzdt_delay = function(u,x,h,p,t)
        Y, Z = u
        Y_delay, Z_delay = h(p, t - p.n)
        X, X_delay = x[1]

        [p.a*p.c*X*(1 - Y - Z) -
            p.a*p.c*X_delay*(1 - Y_delay - Z_delay)*exp(-p.g*p.n) -
            p.g*Y,
        p.a*p.c*X_delay*(1 - Y_delay - Z_delay)*exp(-p.g*p.n) -
            p.g*Z]
    end
    dxdt_delay = function(u,x,h,p,t)
        X, = u
        Z, _ = x[1]
        [p.m*p.a*p.b*Z*(1 - X) - p.r*X]
    end

    mosquito_delay_model = DelayMachine{Float64, 2}(
        1, 2, 1, dzdt_delay, (u,h,p,t) -> [[u[2], h(p,t - p.n)[2]]])
    human_delay_model = DelayMachine{Float64, 2}(
        1, 1, 1, dxdt_delay, (u,h,p,t) -> [[u[1], h(p, t - p.n)[1]]])
    rm_model = oapply(c, [mosquito_delay_model, human_delay_model])

    params = LVector(a = 0.3, b = 0.55, c = 0.15,
        g = 0.1, n = 10, r = 1.0/200, m = 0.5)

    u0_delay = [0.09, .01, 0.3]
    tspan = (0.0, 365.0*5)
    hist(p,t) = u0_delay;

    prob = DDEProblem(rm_model, u0_delay, [], hist, tspan, params)
    alg = MethodOfSteps(Tsit5())
    sol = solve(prob, alg)
    a, b, c, g, n, r, m = params
    R0 = (m*a^2*b*c*exp(-g*n))/(r*g)
    @test isapprox(last(sol)[3], (R0 - 1)/(R0 + (a*c)/g), atol = 1e-3)
end
