using AlgebraicDynamics
using Catlab
using DelayDiffEq
using Test
using ComponentArrays
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
      p = ComponentArray(τ = τ)
    
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
  
        prob = DDEProblem(oapply(d, [mult_delay, add_delay]), u0, x0, hist, (0, 4.0), ComponentArray(τ = 2.0)) 
        sol1 = solve(prob, alg; dtmax = 0.1)
        if delay
          f = (u,h,p,t) -> [ (h(p,t - p.τ)[2] + x0[1]) * h(p,t - p.τ)[1], h(p, t - p.τ)[1] + h(p,t - p.τ)[2] ]
        else
          f = (u,h,p,t) -> [ (u[2] + x0[1]) * h(p,t - p.τ)[1], u[1] + h(p,t - p.τ)[2] ]
        end
        prob = DDEProblem(f, u0, hist, (0, 4.0), ComponentArray(τ = 2.0))
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
  
    @testset "VectorInterface for InstantaneousContinuousMachine" begin
      m = InstantaneousContinuousMachine{Float64,3}(
        1, 3, 1,
        (u, x, p, t) -> u*1.5, # dynamics
        (u, x, p, t) -> u+x[1], # readout
        [1 => 1]
      )
      
      x1 = Float64.([1,2,3])
      u1 = Float64.([4,5,6])
      
      @test readout(m, u1, [x1], nothing, 0) == u1+x1
      @test eval_dynamics(m, u1, [x1], nothing, 0) == u1*1.5
  
      # compare to analytic soln
      prob = ODEProblem(m, u1, [x1], (0.0, 0.05))
      sol = solve(prob, Tsit5())
      
      @test sol.u[end] ≈ [x*exp(1.5*0.05) for x in u1]
  
      # check it composes
      m1 = InstantaneousContinuousMachine{Float64,3}(2, 3, 1,
        (u, x, p, t) -> x[1] + x[2],
        (u,p,t) -> [u])
      m2 = InstantaneousContinuousMachine{Float64, 3}(1, 3, 2,
        (u, x, p, t) -> x[1] + u,
        (u,p,t) -> [u, 2*u])
      m3 = InstantaneousContinuousMachine{Float64, 3}(1, 3, 1,
        (u, x, p, t) -> x[1],
        (u,p,t) -> [u])
      m = oapply(d_big, [m1, m2, m3])
  
      u1 = 1:3; u2 = 4:6; u3 = 7:9;
      x1 = ones(3); x2 = fill(0.5, 3)
  
      @test readout(m, vcat(u1, u2, u3), nothing, 0) == [u1 + u2, u2, u3]
      @test eval_dynamics(m, vcat(u1, u2, u3), [x1, x2], nothing, 0) == vcat(x1, u2 + x1 + 2*u2 + u3, x2)
  
    end
  
    @testset "VectorInterface for InstantaneousDelayMachine" begin
      m = InstantaneousDelayMachine{Float64,3}(
        1, 3, 1, # ninputs, nstates, noutputs
        (u, x, h, p, t) -> -h(p, t-1), # dynamics
        (u, x, h, p, t) -> u+x[1], # readout
        [1 => 1] # output -> input dependency
      )
      
      x1 = Float64.([1,2,3])
      u1 = Float64.([4,5,6])
      hist(p,t) = u1
      
      @test readout(m, u1, [x1], hist, nothing, 0)  == u1+x1
      @test eval_dynamics(m, u1, [x1], hist, nothing, 0) == -u1
      
      prob = DDEProblem(m, u1, [x1], hist, (0.0, 3.0), nothing)
      sol = solve(prob,MethodOfSteps(Tsit5()),abstol=1e-12,reltol=1e-12)
      
      # DiffEq.jl solution
      prob1 = DDEProblem((du,u,h,p,t) -> begin
        x_lag = -h(p, t-1)
        du[1] = x_lag[1]
        du[2] = x_lag[2]
        du[3] = x_lag[3]
      end, u1, hist, (0.0, 3.0), nothing)
      
      sol1 = solve(prob1,MethodOfSteps(Tsit5()),abstol=1e-12,reltol=1e-12)
      
      @test sol.u[end] ≈ sol1.u[end]
      
    end
  end


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

# CPGDynam Integration
######################

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

    params = ComponentArray(a = 0.3, b = 0.55, c = 0.15,
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

