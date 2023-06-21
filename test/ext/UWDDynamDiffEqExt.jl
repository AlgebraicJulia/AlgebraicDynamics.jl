using Test
using AlgebraicDynamics.UWDDynam
using OrdinaryDiffEq, DelayDiffEq
using Catlab

# UWDDynam Integration
######################

const UWD = UndirectedWiringDiagram

@testset "DDE Problems" begin
    # delay differential equation with analytic solution via method of steps
    # dx/dt = -x(t-1) with history x(t) = 10 for t <= 0
    u0 = 10.0
    x0 = [u0]
    p = nothing
    hist(p, t) = [u0] # history
  
    alg = MethodOfSteps(Tsit5())
  
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
    consistent(sol) = all(vcat(sol.(ts)...) .≈ ref_sol.(ts))
  
    @testset "Imperative diagram" begin
      df(u, h, p, t) = -h(p, t - 1.0)
      r = DelayResourceSharer{Float64}(1, df)
      prob = DDEProblem(r, [u0], hist, (0.0, 3.0), p)
      sol = solve(prob, alg,abstol=1e-12,reltol=1e-12)
      @test consistent(sol)
  
      # test oapply with UWD
      d = UWD(1)
      add_box!(d, 1)
      add_junctions!(d, 1)
      set_junction!(d, [1])
      set_junction!(d, [1], outer=true)
  
      r2 = oapply(d, [r])
  
      @test nstates(r) == nstates(r2)
      @test nports(r) == nports(r2)
      @test portmap(r) == portmap(r2)
  
      @test eval_dynamics(r, x0, hist, p, 0.0) == eval_dynamics(r2, x0, hist, p, 0.0)
      @test exposed_states(r, x0) == exposed_states(r2, x0)
  
      prob = DDEProblem(r2, [u0], hist, (0.0, 3.0), p)
      sol = solve(prob, alg,abstol=1e-12,reltol=1e-12)
      @test consistent(sol)
    end
  
    @testset "Relational diagram" begin
      df(u, h, p, t) = -0.5*h(p, t - 1.0)
      r = DelayResourceSharer{Float64}(1, df)
      d = @relation (x,) begin
        f(x)
        g(x)
      end
      r2 = oapply(d, r)
  
      prob = DDEProblem(r2, [u0], hist, (0.0, 3.0), p)
      sol = solve(prob, alg, abstol = 1e-12, reltol=1e-12)
  
      @test eval_dynamics(r2, x0, hist, p, 0.0) == [-10.0]
      @test consistent(sol)
    end
  end
  