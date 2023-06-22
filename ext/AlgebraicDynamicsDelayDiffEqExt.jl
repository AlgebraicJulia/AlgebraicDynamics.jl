module AlgebraicDynamicsDelayDiffEqExt

using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam
using DelayDiffEq
import DelayDiffEq: DDEProblem

"""    DDEProblem(m::DelayMachine, u0::Vector, x::Vector, h::Function, tspan, p=nothing; kwargs...)

Constructs a `DDEProblem` from the vector field defined by `(u,h,p,t) -> m.dynamics(u,x,h,p,t)`, where the exogenous variables are determined by `x` as in `eval_dynamics()`.
"""
DDEProblem(m::DelayMachine, u0::AbstractVector, xs::AbstractVector, hist, tspan, params=nothing; kwargs...) = 
    DDEProblem((u,h,p,t) -> eval_dynamics(m, u, xs, h, p, t), u0, hist, tspan, params; kwargs...)


"""    DDEProblem(r::DelayResourceSharer, u0::Vector, h, tspan)

Constructs a `DDEProblem` from the vector field defined by `(u,h,p,t) -> r.dynamics(u,h,p,t)`.
"""
DDEProblem(r::DelayResourceSharer, u0::AbstractVector, h, tspan, p=nothing; kwargs...) = 
    DDEProblem(UWDDynam.dynamics(r), u0, h, tspan, p; kwargs...)

end