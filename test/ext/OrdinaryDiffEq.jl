using AlgebraicDynamics
using Catlab
using OrdinaryDiffEq
using Test

# DWDDynam Integration
######################

# Define all of the wiring diagrams that will be used
@present C(FreeBiproductCategory) begin 
    A::Ob 
    f::Hom(A, A)
    g::Hom(A, A⊗A)
    h::Hom(A⊗A, A)
end

d_id = @program C (x::A -> f(x))
d12 = @program(C, (x::A) -> f(f(x)))
d_copymerge = @program(C, (x::A) -> [f(x), f(x)])

d_trace = copy(d_id); b, = box_ids(d_trace)
add_wire!(d_trace, (b, 1) => (b, 1))

d_tot = ocompose(d_copymerge, [d12, d_id])

d_big = WiringDiagram([:A, :A], [:A, :A, :A])
b1 = add_box!(d_big, Box(:f, [:A, :A], [:A]))
b2 = add_box!(d_big, Box(:g, [:A], [:A, :A]))
b3 = add_box!(d_big, Box(:h, [:A], [:A]))

bin = input_id(d_big); bout = output_id(d_big)

add_wires!(d_big, Pair[
    (bin, 1) => (b1, 2),
    (bin, 1) => (b2, 1),
    (bin, 2) => (b3, 1),
    (b1, 1) => (bout, 1), 
    (b2, 1) => (bout, 1),
    (b2, 1) => (bout, 2),
    (b2, 2) => (b2, 1),
    (b3, 1) => (b2, 1),
    (b3, 1) => (bout, 3)
])

# CPGDynam Integration
######################

function simulate(f::ContinuousMachine{T}, nsteps::Int, h::Real, u0::Vector, xs=T[]) where T
    trajectory(euler_approx(f, h), u0, xs, nothing, nsteps)
end

function printsim(traj, stepfun, indxfun, shape)
    for u in stepfun(traj)
        pretty_table(reshape(indxfun(u), shape), equal_columns_width=true, noheader=true)
    end
end

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
g2 = migrate!(Catlab.Graphs.Graph(), pg2)

d3 = ocompose(barbell(3), [id(OpenCPortGraph, 3), lob(3)])
pg3 = ocompose(d3, [gl,gm,gr])
g3 = migrate!(Catlab.Graphs.Graph(), pg3)

α₁ = 1
fm = ContinuousMachine{Float64}(4, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 4)))
fl = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fr = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
ft = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fb = ContinuousMachine{Float64}(3, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 3)))
fc = ContinuousMachine{Float64}(2, 1, (u,x,p,t) -> α₁ * (sum(x) .- u .* length(x)), (u,p,t)->collect(repeated(u[1], 2)))

f₁ = oapply(gl, [fc, fl, fc])
f₂ = oapply(gm, [ft, fm, fb])
f₃ = oapply(gr, [fc, fr, fc])

F = oapply(pg3, [fc, fl, fc, ft, fm, fb, fc, fr, fc])
u₀ = zeros(Float64, 9)
u₀[5] = 1.0

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

# Trajectories
##############

const UWD = UndirectedWiringDiagram

approx_equal(u, v) = abs(maximum(u - v)) < .1

tspan = (0.0, 100.0)
tspan_short = (0.0, 10.0)


@testset "Resource sharers" begin
  dx(x) = [1 - x[1]^2, 2*x[1]-x[2]]
  dy(y) = [1 - y[1]^2]
  r = ContinuousResourceSharer{Real}(2, (u,p,t) -> dx(u))

  u0 = [-1.0, -2.0]
  prob = ODEProblem(r, u0, tspan)
  @test prob isa ODEProblem
  sol = solve(prob, Tsit5())
  @test all(s -> s == u0, sol.u)

  u0 = [.1, 10.0]
  prob = ODEProblem(r, u0, tspan)
  sol = solve(prob, Tsit5())
  @test approx_equal(last(sol), [1.0, 2.0])

  h = 0.2
  dr = euler_approx(r, h)
  sol = trajectory(dr, u0, nothing, tspan) # nothing because no parameters
  @test approx_equal(last(sol), [1.0, 2.0])

  dr = DiscreteResourceSharer{Real}(1, (u,p,t) -> dy(u))
  u0 = [1.0]
  sol = trajectory(dr, u0, nothing, tspan)
  @test all(((i,s),) -> s[1] == i%2, enumerate(sol.u))
end

@testset "Machines" begin
  @testset "Euler approximation" begin
    uf(u, x, p, t) = [x[1] - u[1]*x[2]]
    rf(u,p,t) = u
    u0 = [1.0]
    m = ContinuousMachine{Float64}(2,1, 1, uf, rf)

    xs = [t -> 1, t -> 1]
    prob = ODEProblem(m, u0, xs, tspan)
    @test prob isa ODEProblem
    sol = solve(prob, Tsit5())
    @test all(s -> s == u0, sol.u)

    xs = t->t
    prob = ODEProblem(m, u0, xs, tspan)
    sol = solve(prob, Tsit5())
    @test all(s -> s == u0, sol.u)

    xs = [2, 1]
    prob = ODEProblem(m, u0, xs, tspan)
    sol = solve(prob, Tsit5())
    @test approx_equal(last(sol), [2.0])

    h = 0.2
    dm = euler_approx(m,h)
    sol = trajectory(dm, u0, xs, nothing, tspan)
    @test approx_equal(last(sol), [2.0])
  end

  @testset "oapply" begin
    uf(u, x, p, t) = [x[1] - u[1], 0.0]
    rf(u,p,t) = u
    mf = ContinuousMachine{Float64}(2,2,2, uf, rf)
    d_id = singleton_diagram(Box(:f, [:A, :A], [:A, :A]))
    m_id = oapply(d_id, [mf])

    xs = [0.0, 0.0]
    u0 = [10.0, 2.0]
    @testset "" for m in [mf, m_id]
      prob = ODEProblem(m, u0, xs, tspan)
      sol = solve(prob, Tsit5(); dtmax = 1)
      @test approx_equal(last(sol), [0.0, 2.0])
    end
  end
end

@testset "Lotka-Volterra model" begin
  params = [0.3, 0.015, 0.015, 0.7]
  u0 = [10.0, 100.0]

  bounds(sol::ODESolution) = all(s -> all([0, 0] .< s .< [200.0, 150.0]), sol.u)

  @testset "Resource sharer implementation" begin
    dotr(u,p,t) = p[1]*u
    dotrf(u,p,t) = [-p[2]*u[1]*u[2], p[3]*u[1]*u[2]]
    dotf(u,p,t) = -p[4]*u

    r = ContinuousResourceSharer{Real}(1, dotr)
    rf_pred = ContinuousResourceSharer{Real}(2, dotrf)
    f = ContinuousResourceSharer{Real}(1, dotf)

    rf_pattern = UWD(0)
    add_box!(rf_pattern, 1); add_box!(rf_pattern, 2); add_box!(rf_pattern, 1)
    add_junctions!(rf_pattern, 2)
    set_junction!(rf_pattern, [1,1,2,2])

    lv = oapply(rf_pattern, [r, rf_pred, f])

    prob = ODEProblem(lv, u0, tspan, params)
    sol = solve(prob, Tsit5())
    @test bounds(sol)

    h = 0.002
    lv_discrete = oapply(rf_pattern, euler_approx([r, rf_pred, f], h))
    sol2 = trajectory(lv_discrete, u0, params, tspan; dt = h)
    @test bounds(sol2)
  end

  @testset "Machine implementation" begin
    dotr(u, x, p, t) = [p[1]*u[1] - p[2]*u[1]*x[1]]
    dotf(u, x, p, t) = [p[3]*u[1]*x[1] - p[4]*u[1]]

    rmachine = ContinuousMachine{Real}(1,1,1, dotr, (r,p,t) -> r)
    fmachine = ContinuousMachine{Real}(1,1,1, dotf, (f,p,t) -> f)

    rf_pattern = WiringDiagram([],[])
    boxr = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
    boxf = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
    add_wires!(rf_pattern, Pair[
      (boxr, 1) => (boxf, 1),
      (boxf, 1) => (boxr, 1)
    ])

    rf_machine = oapply(rf_pattern, [rmachine, fmachine])
    prob = ODEProblem(rf_machine, u0, tspan, params)
    sol = solve(prob, Tsit5(); dtmax = .1)
    @test bounds(sol)

    h = 0.005
    lv_discrete = oapply(rf_pattern, euler_approx([rmachine, fmachine], h))
    sol = trajectory(lv_discrete, u0, params, tspan_short; dt = h)
    @test bounds(sol)
  end
end

@testset "Ocean model" begin
  α, β, γ, δ, β′, γ′, δ′ = 0.3, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35
  params = [α, β, γ, δ, β′, γ′, δ′]
  u0 = [100.0, 10.0, 2.0]

  bounds(sol::ODESolution) = all(s -> all([0, 0, 0] .< s .< [200.0, 75, 20]), sol.u)

  dotfish(f, x, p, t) = [p[1]*f[1] - p[2]*x[1]*f[1]]
  dotFISH(F, x, p, t) = [p[3]*x[1]*F[1] - p[4]*F[1] - p[5]*x[2]*F[1]]
  dotsharks(s, x, p, t) = [-p[7]*s[1] + p[6]*s[1]*x[1]]

  fish   = ContinuousMachine{Real}(1,1,1, dotfish,   (f,p,t)->f)
  FISH   = ContinuousMachine{Real}(2,1,2, dotFISH,   (F,p,t)->[F[1], F[1]])
  sharks = ContinuousMachine{Real}(1,1,1, dotsharks, (s,p,t)->s)

  ocean_pat = WiringDiagram([], [])
  boxf = add_box!(ocean_pat, Box(nothing, [nothing], [nothing]))
  boxF = add_box!(ocean_pat, Box(nothing, [nothing, nothing], [nothing, nothing]))
  boxs = add_box!(ocean_pat, Box([nothing], [nothing]))
  add_wires!(ocean_pat, Pair[
    (boxf, 1) => (boxF, 1),
    (boxs, 1) => (boxF, 2),
    (boxF, 1) => (boxf, 1),
    (boxF, 2) => (boxs, 1)
  ])

  ocean = oapply(ocean_pat, [fish, FISH, sharks])
  prob = ODEProblem(ocean, u0, tspan, params)
  sol = solve(prob, Tsit5(); dtmax = 0.1)
  @test bounds(sol)

  h = 0.01
  sol = trajectory(euler_approx(ocean, h), u0, params, tspan_short; dt = h)
  @test bounds(sol)
end




