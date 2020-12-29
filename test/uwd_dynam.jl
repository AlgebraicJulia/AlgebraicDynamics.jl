using AlgebraicDynamics.UWDDynam
using Catlab.WiringDiagrams
using Test

const UWD = UndirectedWiringDiagram

@testset "UWDDynam" begin
  dx(x) = [x[1]^2, 2*x[1]-x[2]]

  r = ResourceSharer{Float64}(2, dx)
  #identity
  d = UWD(2)
  add_box!(d, 2)
  add_junctions!(d, 2)
  set_junction!(d, [1,2])
  set_junction!(d, [1,2], outer=true)
  
  r2 = oapply(d, [r])
  @test nstates(r) == nstates(r2)
  @test nports(r) == nports(r2)
  @test portmap(r) == portmap(r2)
  x0 = [10.0, 7.5]
  @test eval_dynamics(r, x0) == eval_dynamics(r2, x0)
  @test exposed_states(r, x0) == exposed_states(r2, x0)

  # merge
  d = UWD(1)
  add_box!(d, 2)
  add_junctions!(d, 1)
  set_junction!(d, [1,1])
  set_junction!(d, [1], outer=true)
  
  r2 = oapply(d, [r])
  @test nstates(r2) == 1
  @test nports(r2) == 1
  @test portmap(r2) == [1]
  @test eval_dynamics(r2, [5.0]) == [30.0]
  @test exposed_states(r2, [5.0]) == [5.0]


  # copy
  r = ResourceSharer{Float64}(1, 2, dx, [2])
  d = UWD(2)
  add_box!(d, 1)
  add_junctions!(d, 1)
  set_junction!(d, [1])
  set_junction!(d, [1,1], outer = true)
  r2 = oapply(d, [r])
  @test nstates(r2) == 2
  @test nports(r2) == 2
  @test portmap(r2) == [2,2]
  @test eval_dynamics(r2, x0) == eval_dynamics(r, x0)
  @test exposed_states(r2, x0) == [x0[2], x0[2]]
  
  # copy states and merge back together
  r = ResourceSharer{Float64}(2, 2, dx, [1,1])
  d = UWD(1)
  add_box!(d, 2)
  add_junctions!(d, 1)
  set_junction!(d, [1,1])
  set_junction!(d, [1], outer = true)
  r2 = oapply(d, [r])
  @test nstates(r2) == 2
  @test nports(r2) == 1
  @test portmap(r2) == [1]
  @test eval_dynamics(r2, x0) == eval_dynamics(r, x0)
  @test exposed_states(r2, x0) == [x0[1]]


  # copy states and merge with otherwise
  dy(y) = [1 - y[1]^2]
  r = ResourceSharer{Float64}(1, dy)
  rcopy = ResourceSharer{Float64}(2, 1, dy, [1,1])
  d = UWD(2)
  add_box!(d, 1); add_box!(d, 2); add_box!(d, 1)
  add_junctions!(d, 2)
  set_junction!(d, [1,1,2,2])
  set_junction!(d, [1,2], outer = true)
  r2 = oapply(d, [r, rcopy, r])
  @test nstates(r2) == 1
  @test nports(r2) == 2
  @test portmap(r2) == [1,1]
  @test eval_dynamics(r2, [7.0]) == [3 *(1 - 7.0^2)]
  @test exposed_states(r2, [7.0]) == [7.0, 7.0]


  # add a state
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

  # lots of boxes
  r = ResourceSharer{Float64}(2, dx)
  s = ResourceSharer{Float64}(1, dy)
  d = UWD(5)
  add_box!(d, 2); add_box!(d, 1); add_box!(d, 2)
  add_junctions!(d, 4)
  set_junction!(d, [1,1,1,4,2])
  set_junction!(d, [1,1,2,3,3], outer = true)
  r2 = oapply(d, [r, s, r])
  @test nstates(r2) == 4
  @test nports(r2) == 5
  @test portmap(r2) == [1,1, 3, 4,4]
  @test eval_dynamics(r2, [2.0, 7.0, 3.0, 5.0]) == [3.0, 49.0, 11.0, 0.0]
  @test exposed_states(r2, [2.0, 7.0, 3.0, 5.0]) == [2.0, 2.0, 3.0, 5.0, 5.0]

  # substitute and oapply commute
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

  s = ResourceSharer{Float64}(1, 2, dx, [2])

  r1 = oapply(dtot, [s,s,r])
  r2 = oapply(d, [s, oapply(din, [s,r])])

  @test nstates(r1) == nstates(r2)
  @test nports(r1) == nports(r2)
  @test portmap(r1) == portmap(r2)
  x0 = [2.0, 3.0, 5.0, 7.0, 11.0]
  @test eval_dynamics(r1, x0) == eval_dynamics(r2, x0)





  
end # test set



# d = UWD(2)
# add_parts!(d, :Box, 3)
# add_parts!(d, :Junction, 2, outer_junction = [1,2])
# add_parts!(d, :Port, 4, box=[1,2,2,3], junction=[1,1,2,2])

# α, β, γ, δ = 0.3, 0.015, 0.015, 0.7

# dotr(x,p,t)  = α*x
# dotrf(x,p,t) = [-β*x[1]*x[2], γ*x[1]*x[2]]
# dotf(x,p,t)  = -δ*x

# rabbit_growth       = ResourceSharer{Float64}(1, 1, dotr,  [1])
# rabbitfox_predation = ResourceSharer{Float64}(2, 2, dotrf, [1,2])
# fox_decline         = ResourceSharer{Float64}(1, 1, dotf,  [1])

# xs = [rabbit_growth, rabbitfox_predation, fox_decline]

# rf = oapply(d, xs)