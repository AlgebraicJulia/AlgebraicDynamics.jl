using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam

using Catlab.WiringDiagrams
using Catlab.CategoricalAlgebra

using StaticArrays
using DynamicalSystems
import DynamicalSystems: DiscreteDynamicalSystem, trajectory

using OrdinaryDiffEq
import OrdinaryDiffEq: ODEProblem

using Base.Iterators

ODEProblem(r::ContinuousResourceSharer, args...) = ODEProblem(r.dynamics, args...)

ODEProblem(m::ContinuousMachine, fs::Vector{T}, args...) where {T<:Function} = begin
  nparams(m) == length(fs) || error("Need a function to fill every exogenous variable")
  ODEProblem((u,p,t) -> m.dynamics(u, [f(t) for f in fs], p, t), args...)
end

ODEProblem(m::ContinuousMachine, f::Function, args...) = 
  ODEProblem(m, collect(repeated(f, nparams(m))), args...)


function oapply(d::WiringDiagram, xs::AbstractDict)
  oapply(d, [xs[box.value] for box in boxes(d)])
end


dx(x) = [1 - x[1]^2, 2*x[1]-x[2]]
dy(y) = [1 - y[1]^2]

r = ContinuousResourceSharer{Float64}(2, (u,p,t) -> dx(u))
u0 = [.1, 10.0]
tspan = (0.0, 100.0)
prob = ODEProblem(r, u0, tspan)
sol = solve(prob, Tsit5())

uf(x,p) = [p[1] - x[1]*p[2]]
rf(x) = x
u0 = [1.0]
m = ContinuousMachine{Float64}(2,1, 1, (u,p,q,t) -> uf(u,p), (u,q,t) -> rf(u))
fs = [t -> 1+t, t -> 2]
prob = ODEProblem(m, fs, [1.0], tspan)
sol = solve(prob, Tsit5())

prob = ODEProblem(m, t -> t^2, [1.0], tspan)
sol = solve(prob, Tsit5())
