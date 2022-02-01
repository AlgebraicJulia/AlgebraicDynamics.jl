using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra

using OrdinaryDiffEq

using Test
const UWD = UndirectedWiringDiagram

approx_equal(u, v) = abs(maximum(u - v)) < .1

tspan = (0.0, 100.0)

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
  dds = DiscreteProblem(dr, u0, tspan, nothing) # nothing because no parameters
  sol = solve(dds, FunctionMap())
  @test approx_equal(last(sol), [1.0, 2.0])

  dr = DiscreteResourceSharer{Real}(1, (u,p,t) -> dy(u))
  u0 = [1.0]
  dds = DiscreteProblem(dr, u0, tspan, nothing)
  sol = solve(dds, FunctionMap())
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
    prob = DiscreteProblem(dm, u0, xs, (0, 100.0), nothing)
    sol = solve(prob, FunctionMap())
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
  bounds(sol) = all(s -> all([0, 0] .< s .< [200.0, 150.0]), sol.u)

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

    h = 0.01
    lv_discrete = oapply(rf_pattern, euler_approx([r, rf_pred, f], h))
    dds = DiscreteProblem(lv_discrete, u0, (0,100.0), params)
    sol2 = solve(dds, FunctionMap(); dt = h)
    @test bounds(sol)
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
    dds = DiscreteProblem(lv_discrete, u0, (0.0, 10.0),params)
    sol = solve(dds, FunctionMap(); dt = h)
    @test bounds(sol)
  end
end

@testset "Ocean model" begin
  α, β, γ, δ, β′, γ′, δ′ = 0.3, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35
  params = [α, β, γ, δ, β′, γ′, δ′]
  u0 = [100.0, 10.0, 2.0]
  bounds(sol) = all(s -> all([0, 0, 0] .< s .< [200.0, 75, 20]), sol.u)

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
  dds = DiscreteProblem(euler_approx(ocean, h), u0, (0.0, 10.0), params)
  sol = solve(dds, FunctionMap(), dt = h)
  @test bounds(sol)
end
