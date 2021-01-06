using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra

using DynamicalSystems
using OrdinaryDiffEq

using Test

approx_equal(u, v) = abs(maximum(u - v)) < .1

# Resource sharer tests
dx(x) = [1 - x[1]^2, 2*x[1]-x[2]]
dy(y) = [1 - y[1]^2]

r = ContinuousResourceSharer{Real}(2, (u,p,t) -> dx(u))

u0 = [-1.0, -2.0]
tspan = (0.0, 100.0)
prob = ODEProblem(r, u0, tspan)
@test prob isa ODEProblem
sol = solve(prob, Tsit5())
for i in 1:length(sol.t)
  @test sol[i] == u0
end

u0 = [.1, 10.0]
prob = ODEProblem(r, u0, tspan)
sol = solve(prob, Tsit5())
@test approx_equal(last(sol), [1.0, 2.0])



h = 0.2
dr = euler_approx(r, h)
dds = DiscreteDynamicalSystem(dr, u0, nothing) # nothing because no parameters
t = trajectory(dds, 100)
s = trajectory(dr, u0, 100, nothing)
@test t==s
@test approx_equal(last(t), [1.0, 2.0])



dr = DiscreteResourceSharer{Real}(1, (u,p,t) -> dy(u))
u0 = [1.0]
dds = DiscreteDynamicalSystem(dr, u0[1], nothing)
t = trajectory(dds, 100)
for i in 1:length(t)
  @test t[i] == i%2
end
s = trajectory(dr, u0, 100, nothing)
@test t == s

dds = DiscreteDynamicalSystem(dr, u0, nothing)
t = trajectory(dds, 100)
for i in 1:length(t)
  @test t[i] == i%2
end
@test t == s

# Machine tests
uf(x,p) = [p[1] - x[1]*p[2]]
rf(x) = x
u0 = [1.0]
m = ContinuousMachine{Any}(2,1, 1, (u,i,p,t) -> uf(u,i), (u,p,t) -> rf(u))
fs = [t -> 1, t -> 1]
prob = ODEProblem(m, fs, u0, tspan)
@test prob isa ODEProblem

sol = solve(prob, Tsit5())
for i in 1:length(sol.t)
  @test sol[i] == u0
end

fs = [t -> t, t -> t]
prob = ODEProblem(m, fs, u0, tspan)
sol = solve(prob, Tsit5())
for i in 1:length(sol.t)
  @test sol[i] == u0
end

fs = [t -> 2, t -> 1]
prob = ODEProblem(m, fs, u0, tspan)
sol = solve(prob, Tsit5())
@test approx_equal(last(sol), [2.0])

dm = euler_approx(m,h)
s = trajectory(dm, fs, u0, 100, nothing)
@test approx_equal([last(s)], [2.0])

dds = DiscreteDynamicalSystem(dm, fs, u0, nothing)
t = trajectory(dds, 100)
@test s==t
