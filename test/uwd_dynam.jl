using AlgebraicDynamics.UWDDynam
using Catlab

using Test

const UWD = UndirectedWiringDiagram

@testset "UWDDynam" begin
  dx(x, p, t) = [x[1]^2, 2*x[1]-x[2]]
  dy(y, p, t) = [1 - y[1]^2]

  r = ContinuousResourceSharer{Float64}(2, dx)
  s = ContinuousResourceSharer{Float64}(1, dy)
  t = ContinuousResourceSharer{Float64}(1, 2, dx, [2])
  @test all(eltype.([r, s, t]) .== Float64)

  x0 = [10.0, 7.5]

  @testset "Identity & merge" begin
    d = @relation (x,y) begin
      f(x,y)
    end
    r2 = oapply(d, [r])

    @test nstates(r) == nstates(r2)
    @test nports(r) == nports(r2)
    @test portmap(r) == portmap(r2)
    @test eval_dynamics(r, x0) == eval_dynamics(r2, x0)
    @test exposed_states(r, x0) == exposed_states(r2, x0)
    @test ports(r) == [1, 2]
    @test ports(r2) == [:x, :y]

    h = 0.1
    drs = oapply(d, [euler_approx(r, h)])
    drs2 = euler_approx(r2, h)
    @test eval_dynamics(drs, x0) == eval_dynamics(drs2, x0)
    drs3 = euler_approx(r2)
    drs4 = oapply(d, [euler_approx(r)])
    @test eval_dynamics(drs, x0) == eval_dynamics(drs3, x0, [h])
    @test eval_dynamics(drs, x0) == eval_dynamics(drs4, x0, [h])

    d = @relation (x,) begin
      f(x,x)
    end
    r2 = oapply(d, [r])

    @test nstates(r2) == 1
    @test nports(r2) == 1
    @test portmap(r2) == [1]
    @test eval_dynamics(r2, [5.0]) == [30.0]
    @test exposed_states(r2, [5.0]) == [5.0]
    @test ports(r2) == [:x]

    let r = ContinuousResourceSharer{Float64}(2, (u,p,t) -> [u[1]*p[1], u[2]*p[2] + t])
    r2 = oapply(d, [r])
    @test eval_dynamics(r2, [5.0], [1.0, 2.0], 10.0) == [25.0]
    end
  end

  @testset "Copy" begin
    d = UWD(2)
    add_box!(d, 1)
    add_junctions!(d, 1)
    set_junction!(d, [1])
    set_junction!(d, [1,1], outer = true)

    let r = t
    r2 = oapply(d, r)
    @test nstates(r2) == 2
    @test nports(r2) == 2
    @test portmap(r2) == [2,2]
    @test eval_dynamics(r2, x0) == eval_dynamics(r, x0)
    @test exposed_states(r2, x0) == [x0[2], x0[2]]
    end
  end
  
  @testset "Copy & merge" begin
    d = @relation (x,) begin
      f(x,x)
    end

    let r = ContinuousResourceSharer{Float64}(2, 2, dx, [1,1])
    r2 = oapply(d, r)
    @test nstates(r2) == 2
    @test nports(r2) == 1
    @test ports(r2) == [:x]
    @test portmap(r2) == [1]
    @test eval_dynamics(r2, x0) == eval_dynamics(r, x0)
    @test exposed_states(r2, x0) == [x0[1]]
    end
  end

  @testset "Duplicate & add state" begin
    let r = s
    rcopy = ContinuousResourceSharer{Float64}(2, 1, dy, [1,1])
    d = HypergraphDiagram{Nothing, Symbol}(2)
    add_box!(d, 1, name = :r); add_box!(d, 2, name = :copy); add_box!(d, 1, name = :r)
    add_junctions!(d, 2)
    set_junction!(d, [1,1,2,2])
    set_junction!(d, [1,2], outer = true)
    xs = Dict(:r => r, :copy => rcopy)
    r2 = oapply(d, xs)
    @test nstates(r2) == 1
    @test nports(r2) == 2
    @test portmap(r2) == [1,1]
    @test eval_dynamics(r2, [7.0]) == [3 *(1 - 7.0^2)]
    @test exposed_states(r2, [7.0]) == [7.0, 7.0]

    d = UWD(2)
    add_box!(d, 1)
    add_junctions!(d, 2)
    set_junction!(d, [1])
    set_junction!(d, [1,2], outer = true)
    r2 = oapply(d, [r])
    @test nstates(r2) == 2
    @test nports(r2) == 2
    @test portmap(r2) == [1,2]
    @test eval_dynamics(r2, [7.0, 11.0]) == [-48.0, 0.0]
    @test exposed_states(r2, [7.0, 11.0]) == [7.0, 11.0]
    end
  end

  @testset "Boxes" begin
    xs = Dict(:r => r, :s => s)
    d = HypergraphDiagram{Nothing, Symbol}(5)
    add_box!(d, 2, name = :r); add_box!(d, 1, name = :s); add_box!(d, 2, name = :r)
    add_junctions!(d, 4)
    set_junction!(d, [1,1,1,4,2])
    set_junction!(d, [1,1,2,3,3], outer = true)
    r2 = oapply(d, xs)
    @test nstates(r2) == 4
    @test nports(r2) == 5
    @test portmap(r2) == [1,1, 3, 4,4]
    x0 = [2.0, 7.0, 3.0, 5.0]
    @test eval_dynamics(r2, x0) == [3.0, 49.0, 11.0, 0.0]
    @test exposed_states(r2, x0) == [2.0, 2.0, 3.0, 5.0, 5.0]

    h = 0.1
    dr = oapply(d, euler_approx([r,s,r], h))
    dr2 = euler_approx(r2, h)
    dr3 = oapply(d, euler_approx([r,s,r]))
    dr4 = euler_approx(r2)
    @test eval_dynamics(dr, x0) == [2.3, 11.9, 4.1, 5.0]
    @test eval_dynamics(dr, x0) == eval_dynamics(dr2, x0)
    @test eval_dynamics(dr, x0) == eval_dynamics(dr3, x0, [h], 0)
    @test eval_dynamics(dr, x0) == eval_dynamics(dr4, x0, [h], 0)

    h = 0.25
    dr = oapply(d, euler_approx(xs, h))
    dr2 = euler_approx(r2, h)
    dr3 = oapply(d, euler_approx(xs))
    dr4 = euler_approx(r2)
    @test eval_dynamics(dr, x0) == eval_dynamics(dr2, x0)
    @test eval_dynamics(dr, x0) == eval_dynamics(dr3, x0, [h], 0)
    @test eval_dynamics(dr, x0) == eval_dynamics(dr4, x0, [h], 0)
  end

  @testset "Commutativity: substitute & oapply" begin
    d = UWD(4)
    add_box!(d, 1); add_box!(d, 2)
    add_junctions!(d, 3)
    set_junction!(d, [1,2,1])
    set_junction!(d, [1,2,3,3], outer = true)

    din = UWD(2)
    add_box!(din, 1); add_box!(din, 2)
    add_junctions!(din, 2)
    set_junction!(din, [1,1,2])
    set_junction!(din, [1,2], outer = true)

    dtot = ocompose(d, 2, din)

    let s = t
    r1 = oapply(dtot, [s,s,r])
    r2 = oapply(d, [s, oapply(din, [s,r])])

    @test nstates(r1) == nstates(r2)
    @test nports(r1) == nports(r2)
    @test portmap(r1) == portmap(r2)
    x0 = [2.0, 3.0, 5.0, 7.0, 11.0]
    @test eval_dynamics(r1, x0) == eval_dynamics(r2, x0)
    end
  end
end
