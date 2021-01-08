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
s = trajectory(dm, fs, u0, nothing, 100)
@test approx_equal([last(s)], [2.0])

dds = DiscreteDynamicalSystem(dm, fs, u0, nothing)
t = trajectory(dds, 100)
@test s==t

# machines - oapply
uf(x, p, q, t) = [p[1] - x[1], 0.0]
rf(x, q, t) = x
mf = ContinuousMachine{Float64}(2,2,2, uf, rf)

d_id = singleton_diagram(Box(:f, [:A, :A], [:A, :A]))
m_id = oapply(d_id, [mf])
fs = [t -> 0, t-> 0]
u0 = [10.0, 2.0]
prob1 = ODEProblem(mf, fs, u0, tspan)
prob2 = ODEProblem(m_id, fs, u0, tspan)

sol1 = solve(prob1, Tsit5(); dtmax = 1)
sol2 = solve(prob2, Tsit5(); dtmax = 1)
@test approx_equal(last(sol1), [0.0, 2.0])
@test approx_equal(last(sol2), [0.0, 2.0])

# Lokta Volterra models

# as resource sharers
α, β, γ, δ = 0.3, 0.015, 0.015, 0.7
dotr(x,p,t) = α*x
dotrf(x,p,t) = [-β*x[1]*x[2], γ*x[1]*x[2]]
dotf(x,p,t) = -δ*x

r = ContinuousResourceSharer{Real}(1, dotr)
rf_pred = ContinuousResourceSharer{Real}(2, dotrf)
f = ContinuousResourceSharer{Real}(1, dotf)


rf_pattern = UWD(0)
add_box!(rf_pattern, 1); add_box!(rf_pattern, 2); add_box!(rf_pattern, 1)
add_junctions!(rf_pattern, 2)
set_junction!(rf_pattern, [1,1,2,2])

lv = oapply(rf_pattern, [r, rf_pred, f])

u0 = [10.0, 100.0]
prob = ODEProblem(lv, u0, tspan)
sol = solve(prob, Tsit5())

for i in 1:length(sol.t)
  @test ( 0 < sol[i][1] < 200.0 ) && (0 < sol[i][2] < 150.0) 
end

lv_discrete = oapply(rf_pattern, euler_approx([r, rf_pred, f], h))
dds = DiscreteDynamicalSystem(lv_discrete, u0, nothing)
@test trajectory(dds, 100) == trajectory(lv_discrete, u0, nothing, 100)

# as machines

α, β, γ, δ = 0.3, 0.015, 0.015, 0.7

dotr(x, p, q, t) = [α*x[1] - β*x[1]*p[1]]
dotf(x, p, q, t) = [γ*x[1]*p[1] - δ*x[1]]

rmachine = ContinuousMachine{Real}(1,1,1, dotr, (r,q,t) -> r)
fmachine = ContinuousMachine{Real}(1,1,1, dotf, (f,q,t) -> f)

rf_pattern = WiringDiagram([],[])
boxr = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
boxf = add_box!(rf_pattern, Box(nothing, [nothing], [nothing]))
add_wires!(rf_pattern, Pair[
  (boxr, 1) => (boxf, 1), 
  (boxf, 1) => (boxr, 1)
])

u0 = [10.0, 100.0]
rf_machine = oapply(rf_pattern, [rmachine, fmachine])
prob = ODEProblem(rf_machine, [], u0, tspan)
sol = solve(prob, Tsit5(); dtmax = .1)
for i in 1:length(sol.t)
  @test ( 0 < sol[i][1] < 200.0 ) && (0 < sol[i][2] < 150.0) 
end

lv_discrete = oapply(rf_pattern, euler_approx([rmachine, fmachine], h))
dds = DiscreteDynamicalSystem(lv_discrete, [], u0, nothing)
t = trajectory(dds, 100)
s = trajectory(lv_discrete, [], u0, nothing, 100)
@test t == s


# ocean
α, β, γ, δ, β′, γ′, δ′ = 0.3, 0.015, 0.015, 0.7, 0.017, 0.017, 0.35

dotfish(f, x, p, t) = [α*f[1] - β*x[1]*f[1]]
dotFISH(F, x, p, t) = [γ*x[1]*F[1] - δ*F[1] - β′*x[2]*F[1]]
dotsharks(s, x, p, t) = [-δ′*s[1] + γ′*s[1]*x[1]]

fish   = ContinuousMachine{Real}(1,1,1, dotfish,   (f, p, t) ->f)
FISH   = ContinuousMachine{Real}(2,1,2, dotFISH,   (F, p, t)->[F[1], F[1]])
sharks = ContinuousMachine{Real}(1,1,1, dotsharks, (s, p, t) ->s)

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
prob = ODEProblem(ocean, [], u0, tspan)
sol = solve(prob, Tsit5(); dtmax = 0.1) #getting runaway solution and idk why. The eval_dynamics for ocean looks good
for i in 1:length(sol.t)
  @test 0 < sol[i][1] < 200.0 && (0 < sol[i][2] < 75) && (0 < sol[i][3] < 20)
end

dds = DiscreteDynamicalSystem(euler_approx(ocean, h), [], u0, nothing)
t = trajectory(dds, 100)
s = trajectory(euler_approx(ocean, h), [], u0, nothing, 100)
@test s == t
