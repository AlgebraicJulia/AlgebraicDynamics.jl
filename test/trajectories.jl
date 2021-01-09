using AlgebraicDynamics
using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra

using DynamicalSystems
using OrdinaryDiffEq

using Test
const UWD = UndirectedWiringDiagram

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
s = trajectory(dr, u0, nothing, 100)
@test t==s
@test approx_equal(last(t), [1.0, 2.0])



dr = DiscreteResourceSharer{Real}(1, (u,p,t) -> dy(u))
u0 = [1.0]
dds = DiscreteDynamicalSystem(dr, u0[1], nothing)
t = trajectory(dds, 100)
for i in 1:length(t)
  @test t[i] == i%2
end
s = trajectory(dr, u0, nothing, 100)
@test t == s

dds = DiscreteDynamicalSystem(dr, u0, nothing)
t = trajectory(dds, 100)
for i in 1:length(t)
  @test t[i] == i%2
end
@test t == s

s = trajectory(dr, u0, nothing, 100; dt = 2)
for i in 1:length(s)
  @test s[i] == 1.0
end


# Machine tests
uf(u, x, p, t) = [x[1] - u[1]*x[2]]
rf(u) = u
u0 = [1.0]
m = ContinuousMachine{Any}(2,1, 1, uf, rf)
xs = [t -> 1, t -> 1]
prob = ODEProblem(m, u0, xs, tspan)
@test prob isa ODEProblem

sol = solve(prob, Tsit5())
for i in 1:length(sol.t)
  @test sol[i] == u0
end

xs = t->t
prob = ODEProblem(m, u0, xs, tspan)
sol = solve(prob, Tsit5())
for i in 1:length(sol.t)
  @test sol[i] == u0
end

xs = [2, 1]
prob = ODEProblem(m, u0, xs, tspan)
sol = solve(prob, Tsit5())
@test approx_equal(last(sol), [2.0])

dm = euler_approx(m,h)
s = trajectory(dm, u0, xs, nothing, 100)
@test approx_equal([last(s)], [2.0])

dds = DiscreteDynamicalSystem(dm, u0, xs, nothing)
t = trajectory(dds, 100)
@test s==t

# machines - oapply
uf(u, x, p, t) = [x[1] - u[1], 0.0]
rf(u) = u
mf = ContinuousMachine{Float64}(2,2,2, uf, rf)

d_id = singleton_diagram(Box(:f, [:A, :A], [:A, :A]))
m_id = oapply(d_id, [mf])
xs = [0.0, 0.0]
u0 = [10.0, 2.0]
prob1 = ODEProblem(mf, u0, xs, tspan)
prob2 = ODEProblem(m_id, u0, xs, tspan)

sol1 = solve(prob1, Tsit5(); dtmax = 1)
sol2 = solve(prob2, Tsit5(); dtmax = 1)
@test approx_equal(last(sol1), [0.0, 2.0])
@test approx_equal(last(sol2), [0.0, 2.0])

# Lokta Volterra models

# as resource sharers
params = [0.3, 0.015, 0.015, 0.7]
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

params = [0.3, 0.015, 0.015, 0.7]
u0 = [10.0, 100.0]
prob = ODEProblem(lv, u0, tspan, params)
sol = solve(prob, Tsit5())

for i in 1:length(sol.t)
  @test ( 0 < sol[i][1] < 200.0 ) && (0 < sol[i][2] < 150.0) 
end

lv_discrete = oapply(rf_pattern, euler_approx([r, rf_pred, f], h))
dds = DiscreteDynamicalSystem(lv_discrete, u0, params)
@test trajectory(dds, 100) == trajectory(lv_discrete, u0, params, 100)

# as machines


dotr(u, x, p, t) = [p[1]*u[1] - p[2]*u[1]*x[1]]
dotf(u, x, p, t) = [p[3]*u[1]*x[1] - p[4]*u[1]]

rmachine = ContinuousMachine{Real}(1,1,1, dotr, r -> r)
fmachine = ContinuousMachine{Real}(1,1,1, dotf, f -> f)

rf_pattern = WiringDiagram([],[])
boxr = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
boxf = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
add_wires!(rf_pattern, Pair[
  (boxr, 1) => (boxf, 1), 
  (boxf, 1) => (boxr, 1)
])

u0 = [10.0, 100.0]
params = [0.3, 0.015, 0.015, 0.7]

rf_machine = oapply(rf_pattern, [rmachine, fmachine])
prob = ODEProblem(rf_machine, u0, tspan, params)
sol = solve(prob, Tsit5(); dtmax = .1)
for i in 1:length(sol.t)
  @test ( 0 < sol[i][1] < 200.0 ) && (0 < sol[i][2] < 150.0) 
end

lv_discrete = oapply(rf_pattern, euler_approx([rmachine, fmachine], h))
dds = DiscreteDynamicalSystem(lv_discrete,  u0, params)
t = trajectory(dds, 100)
s = trajectory(lv_discrete, u0, params, 100)
@test t == s


# ocean
α, β, γ, δ, β′, γ′, δ′ = 0.3, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35
params = [α, β, γ, δ, β′, γ′, δ′]

dotfish(f, x, p, t) = [p[1]*f[1] - p[2]*x[1]*f[1]]
dotFISH(F, x, p, t) = [p[3]*x[1]*F[1] - p[4]*F[1] - p[5]*x[2]*F[1]]
dotsharks(s, x, p, t) = [-p[7]*s[1] + p[6]*s[1]*x[1]]

fish   = ContinuousMachine{Real}(1,1,1, dotfish,   f ->f)
FISH   = ContinuousMachine{Real}(2,1,2, dotFISH,   F->[F[1], F[1]])
sharks = ContinuousMachine{Real}(1,1,1, dotsharks, s->s)

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

u0 = [100.0, 10.0, 2.0]
ocean = oapply(ocean_pat, [fish, FISH, sharks])
prob = ODEProblem(ocean, u0, tspan, params)
sol = solve(prob, Tsit5(); dtmax = 0.1)
for i in 1:length(sol.t)
  @test 0 < sol[i][1] < 200.0 && (0 < sol[i][2] < 75) && (0 < sol[i][3] < 20)
end

dds = DiscreteDynamicalSystem(euler_approx(ocean, h), u0, params)
t = trajectory(dds, 100)
s = trajectory(euler_approx(ocean, h), u0, params, 100)
@test s == t
