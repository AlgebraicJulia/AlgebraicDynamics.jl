using AlgebraicDynamics.DWDDynam
using Catlab.WiringDiagrams, Catlab.Present, Catlab.Theories, Catlab.Programs
using LabelledArrays
using DelayDiffEq
using Test

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

@testset "ODE Problems" begin
  uf(u, x, p, t) = [x[1] - u[1]]
  rf(u, args...) = u
  mf = ContinuousMachine{Float64}(1,1,1, uf, rf)
  m_id = oapply(d_id, [mf])
  m12 = oapply(d12, [mf, mf])

  @testset "Identity" begin
    x0 = 1
    p0 = 200
    @test eval_dynamics(m_id, [x0], [p0]) == [p0 - x0]
    @test readout(m_id, [x0]) == [x0]
  end

  x0 = -1
  y0 = 29
  p0 = 200

  @testset "Unfed parameter" begin
    @test eval_dynamics(m12, [x0, y0], [p0]) == [p0 - x0, x0 - y0]
    @test readout(m12, [x0,y0]) == [y0]
  end

  @testset "Copy & merge" begin
    m = oapply(d_copymerge, Dict(:f => mf))
    @test eval_dynamics(m, [x0, y0], [p0]) == [p0 - x0, p0 - y0]
    @test readout(m, [x0,y0]) == [x0 + y0]
    @test ninputs(m) == 1
    @test nstates(m) == 2
    @test noutputs(m) == 1
  end

  @testset "Trace" begin
    m_trace = oapply(d_trace, mf)
    @test nstates(m_trace) == 1
    @test eval_dynamics(m_trace, [x0], [p0]) == [p0]
    @test readout(m_trace, [x0]) == [x0]
  end

  @testset "oapply & ocompose" begin
    m_tot1 = oapply(d_tot, mf)
    m_tot2 = oapply(d_copymerge, [m12, m_id])
    x0 = [-1, 5.5, 20]
    @test nstates(m_tot1) == 3
    @test nstates(m_tot2) == 3

    @test eval_dynamics(m_tot1, x0, [p0]) == eval_dynamics(m_tot2, x0, [p0])
    @test eval_dynamics(m_tot1, x0, [p0]) == [p0 - x0[1], x0[1] - x0[2], p0 - x0[3]]
    @test readout(m_tot1, x0) == [x0[2] + x0[3]]
    @test readout(m_tot2, x0) == [x0[2] + x0[3]]
  end

  @testset "Big diagram" begin
    m1 = ContinuousMachine{Float64}(2,1,1,
            (u, x, p, t) -> [x[1] * x[2]  - u[1]],
            (u,p,t) -> 2*u)
    m2 = ContinuousMachine{Float64}(1, 2, 2,
            (u, x, p, t) -> [u[1]*u[2], x[1]^2*u[2]],
            (u,p,t) -> u)
    m3 = ContinuousMachine{Float64}(1,2,1,
            (u, x, p, t) -> [u[1]^2 - x[1], u[2] - u[1]],
            (u,p,t) -> [u[1] + u[2]])
    xs = Dict(:f => m1, :g => m2, :h => m3, :j => mf)
    m = oapply(d_big, xs)
    @test ninputs(m) == 2
    @test nstates(m) == 5
    @test noutputs(m) == 3

    u = [2.0, 3.0, 5.0, -7.0, -0.5]
    x = [0.1, 11.0]
    @test eval_dynamics(m, u, x) ≈
    [-u[1], u[2]*u[3], (x[1]+u[3]+u[4]+u[5])^2*u[3], u[4]^2 - x[2], u[5] - u[4]]
    @test readout(m, u) == [2*u[1] + u[2], u[2], u[4] + u[5]]
    @test readout(oapply(d_id, m3), [u[4], u[5]]) == [u[4] + u[5]]

      @testset "Euler" for (h, xs) in [(0.15, [m1, m2, m3]),
                                       (0.02, Dict(:h => m3, :f => m1, :g => m2))]
      euler_m = oapply(d_big, euler_approx(xs, h))
      @test eval_dynamics(euler_m, u, x) == u + h*eval_dynamics(m, u, x)
      @test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_approx(m, h), u, x)
      euler_m2 = oapply(d_big, euler_approx(xs))
      @test eval_dynamics(euler_m, u, x) == eval_dynamics(euler_m2, u, x, [h])
    end
  end
end

@testset "DDE Problems" begin
  rf(u,h,p,t) = u
  delay_rf(u,h,p,t) = h(p, t - p.τ)
  alg = MethodOfSteps(Tsit5())

  @testset "Copying" begin
    df(u,x,h,p,t) = h(p, t - p.τ)
    delay = DelayMachine{Float64}(1,1,1,df,rf)
    delay_copy = oapply(d_id, delay)

    hist(p,t) = [0.0]
    u0 = [2.7]
    x0 = [10.0]
    τ = 10.0
    p = LVector(τ = τ)

    @test eval_dynamics(delay, u0, x0, hist, p) == [0.0]
    @test eval_dynamics(delay_copy, u0, x0, hist, p) == [0.0]

    prob = DDEProblem(delay, u0, x0, hist, (0.0, τ - 0.1), p)
    @test last(solve(prob, alg)) == u0

    prob = DDEProblem(delay, u0, x0, hist, (0.0, 2*τ), p)
    prob_copy = DDEProblem(delay_copy, u0, x0, hist, (0.0, 2*τ), p)

    @test last(solve(prob, alg)) == last(solve(prob_copy, alg))
  end

  @testset "Consistency" begin
    # delay differential equation with analytic solution via method of steps
    # dx/dt = -x(t-1) with history x(t) = 10 for t <= 0
    df(u,x,h,p,t) = -h(p, t - 1.0) # dynamics
    delay_machine = DelayMachine{Float64}(0,1,0,df,(u,h,p,t) -> u)

    u0 = 10.0
    p = nothing
    hist(p, t) = [u0] # history

    prob = DDEProblem(delay_machine, [u0], [], hist, (0.0, 3.0), p)
    sol = solve(prob, alg,abstol=1e-12,reltol=1e-12)

    # solve over 0<t<1; we have x(t-1)=10 over this interval therefore
    # dx/dt = -10; dx = -10dt; \int{dx} = \int{-10dt}; x + c1 = -10t + c2
    # x(t) = c - 10t; x(t) = 10 - 10t
    #
    # solve over 1<t<2; we have x(t-1)=10-10(t-1) over this interval therefore
    # dx/dt = -(10 - 10(t-1)); \int{dx} = \int{-(10 - 10(t-1))dt};
    # x + c1 = -10(t-1) + 10\int{(t-1) dt}; x + c1 = -10t + 10((t-1)^2/2) + c2;
    # x(t) = -10(t-1) + 5(t-1)^2
    ref_sol(t) = t < 1 ? 10.0 - 10.0*t : -10.0*(t-1.0) + 5.0*(t-1.0)^2
    ts = [0.5, 0.9, 1.5, 1.9]
    @test all(vcat(sol.(ts)...) .≈ ref_sol.(ts))
  end

  @testset "Composites" begin
    d = ocompose(d_trace, [d12])
    df(u,x,h,p,t) = x[1]*h(p, t - p.τ)
    ef(u,x,h,p,t) = x + h(p, t - p.τ)

    hist(p,t) = [0.0, 0.0]
    u0 = [1.0, 1.0]
    x0 = [1.0]

    @testset "Readout $(delay ? "w/" : "w/o") delay" for delay in [false, true]
      mult_delay = DelayMachine{Float64}(1,1,1, df, delay ? delay_rf : rf)
      add_delay = DelayMachine{Float64}(1,1,1, ef, delay ? delay_rf : rf)

      prob = DDEProblem(oapply(d, [mult_delay, add_delay]), u0, x0, hist, (0, 4.0), LVector(τ = 2.0))
      sol1 = solve(prob, alg; dtmax = 0.1)
      if delay
        f = (u,h,p,t) -> [ (h(p,t - p.τ)[2] + x0[1]) * h(p,t - p.τ)[1], h(p, t - p.τ)[1] + h(p,t - p.τ)[2] ]
      else
        f = (u,h,p,t) -> [ (u[2] + x0[1]) * h(p,t - p.τ)[1], u[1] + h(p,t - p.τ)[2] ]
      end
      prob = DDEProblem(f, u0, hist, (0, 4.0), LVector(τ = 2.0))
      sol2 = solve(prob, alg; dtmax = 0.1)
      @test last(sol1) == last(sol2)
    end
  end

  @testset "VectorInterface" begin
    m1 = ContinuousMachine{Float64,3}(2, 3, 1,
            (u, x, p, t) -> x[1] + x[2],
            (u,p,t) -> [u])
    m2 = ContinuousMachine{Float64, 3}(1, 3, 2,
            (u, x, p, t) -> x[1] + u,
            (u,p,t) -> [u, 2*u])
    m3 = ContinuousMachine{Float64, 3}(1, 3, 1,
            (u, x, p, t) -> x[1],
            (u,p,t) -> [u])
    m = oapply(d_big, [m1, m2, m3])

    u1 = 1:3; u2 = 4:6; u3 = 7:9;
    x1 = ones(3); x2 = fill(0.5, 3)

    @test readout(m, vcat(u1, u2, u3), nothing, 0) == [u1 + u2, u2, u3]
    @test eval_dynamics(m, vcat(u1, u2, u3), [x1, x2], nothing, 0) == vcat(x1, u2 + x1 + 2*u2 + u3, x2)
  end
end
