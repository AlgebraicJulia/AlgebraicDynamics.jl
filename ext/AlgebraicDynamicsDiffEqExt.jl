module AlgebraicDynamicsDiffEqExt

using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.CPortGraphDynam
import AlgebraicDynamics.UWDDynam: trajectory
using OrdinaryDiffEq, DelayDiffEq
import OrdinaryDiffEq: ODEProblem, DiscreteProblem
import DelayDiffEq: DDEProblem

# DWDDynam Integration
######################

"""    ODEProblem(m::ContinuousMachine, x::Vector, u0::Vector, tspan, p=nothing; kwargs...)

Constructs an `ODEProblem` from the vector field defined by `(u,p,t) -> m.dynamics(u,x,p,t)`, where the exogenous variables are determined by `x` as in `eval_dynamics()`.
"""
ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, xs::AbstractVector, tspan, p=nothing; kwargs...)  where T =
    ODEProblem((u,p,t) -> eval_dynamics(m, u, xs, p, t), u0, tspan, p; kwargs...)
  
ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, x::Union{T, Function}, tspan, p=nothing; kwargs...) where T =
    ODEProblem(m, u0, collect(repeated(x, ninputs(m))), tspan, p; kwargs...)

ODEProblem(m::ContinuousMachine{T}, u0::AbstractVector, tspan, p=nothing; kwargs...) where T =
    ODEProblem(m, u0, T[], tspan, p; kwargs...)

"""    DDEProblem(m::DelayMachine, u0::Vector, x::Vector, h::Function, tspan, p=nothing; kwargs...)

Constructs a `DDEProblem` from the vector field defined by `(u,h,p,t) -> m.dynamics(u,x,h,p,t)`, where the exogenous variables are determined by `x` as in `eval_dynamics()`.
"""
DDEProblem(m::DelayMachine, u0::AbstractVector, xs::AbstractVector, hist, tspan, params=nothing; kwargs...) = 
    DDEProblem((u,h,p,t) -> eval_dynamics(m, u, xs, h, p, t), u0, hist, tspan, params; kwargs...)


"""    DiscreteProblem(m::DiscreteMachine, x::Vector, u0::Vector, tspan, p=nothing; kwargs...)

Constructs a `DiscreteProblem` from the equation of motion defined by
`(u,p,t) -> m.dynamics(u,x,p,t)`, where the exogenous variables are determined by `x` as in `eval_dynamics()`.
Pass `nothing` in place of `p` if your system does not have parameters.
"""
DiscreteProblem(m::DiscreteMachine, u0::AbstractVector, xs::AbstractVector, tspan, p; kwargs...) =
    DiscreteProblem((u,p,t) -> eval_dynamics(m, u, xs, p, t), u0, tspan, p; kwargs...)

DiscreteProblem(m::DiscreteMachine, u0::AbstractVector, x, tspan, p; kwargs...) =
  DiscreteProblem(m, u0, collect(repeated(x, ninputs(m))), tspan, p; kwargs...)

DiscreteProblem(m::DiscreteMachine{T}, u0, tspan, p; kwargs...) where T =
    DiscreteProblem(m, u0, T[], tspan, p; kwargs...)

"""    trajectory(m::DiscreteMachine, u0::AbstractVector, x::AbstractVector, p, nsteps::Int; dt::Int = 1)
    trajectory(m::DiscreteMachine, u0::AbstractVector, x::AbstractVector, p, tspan::Tuple{T,T}; dt::T= one(T)) where {T<:Real}

Evolves the machine `m`, for `nsteps` times or over `tspan`, with step size `dt`, initial condition `u0` and parameters `p`.
Any inputs to `m` are determined by `x` as in `eval_dynamics()`. If `m` has no inputs, then you can omit `x`.
"""
trajectory(m::DiscreteMachine, u0::AbstractVector, p, T::Int; dt::Int= 1) =
    trajectory(m, u0, p, (0, T); dt)

trajectory(m::DiscreteMachine, u0::AbstractVector, xs, p, T::Int; dt::Int= 1) =
    trajectory(m, u0, xs, p, (0, T); dt)

trajectory(m::DiscreteMachine, u0::AbstractVector, p, tspan::T; dt::T= one(T)) where {T<:Real} =
    trajectory(m, u0, p, (zero(tspan), tspan); dt)

trajectory(m::DiscreteMachine, u0::AbstractVector, xs, p, tspan::T; dt::T= one(T)) where {T<:Real} =
    trajectory(m, u0, xs, p, (zero(tspan), tspan); dt)

function trajectory(m::DiscreteMachine, u0::AbstractVector, p, tspan::Tuple{T,T}; dt::T= one(T)) where {T<:Real}
  prob = DiscreteProblem(m, u0, tspan, p)
  solve(prob, FunctionMap(); dt = dt)
end

function trajectory(m::DiscreteMachine, u0::AbstractVector, xs, p, tspan::Tuple{T,T}; dt::T= one(T)) where {T<:Real}
  prob = DiscreteProblem(m, u0, xs, tspan, p)
  solve(prob, FunctionMap(); dt = dt)
end

# UWDDynam Integration
######################

"""    ODEProblem(r::ContinuousResourceSharer, u0::Vector, tspan)

Constructs an `ODEProblem` from the vector field defined by `(u,p,t) -> r.dynamics(u,p,t)`.
"""
ODEProblem(r::ContinuousResourceSharer, u0::AbstractVector, tspan, p=nothing; kwargs...) =
    ODEProblem(dynamics(r), u0, tspan, p; kwargs...)

"""    DDEProblem(r::DelayResourceSharer, u0::Vector, h, tspan)

Constructs a `DDEProblem` from the vector field defined by `(u,h,p,t) -> r.dynamics(u,h,p,t)`.
"""
DDEProblem(r::DelayResourceSharer, u0::AbstractVector, h, tspan, p=nothing; kwargs...) = 
    DDEProblem(dynamics(r), u0, h, tspan, p; kwargs...)
    
"""    DiscreteProblem(r::DiscreteResourceSharer, u0::Vector, p)

Constructs a `DiscreteProblem` from the equation of motion defined by `(u,p,t) -> r.dynamics(u,p,t)`.  Pass `nothing` in place of `p` if your system does not have parameters.
"""
DiscreteProblem(r::DiscreteResourceSharer, u0::AbstractVector, tspan, p=nothing; kwargs...) =
    DiscreteProblem(dynamics(r), u0, tspan, p; kwargs...)

"""    trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, nsteps::Int; dt::Int = 1)
    trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, tspan::Tuple{T,T}; dt::T= one(T)) where {T<:Real}

Evolves the resouce sharer `r`, for `nsteps` times or over `tspan`, with step size `dt`, initial condition `u0` and parameters `p`.
"""
trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, T::Int; dt::Int= 1) =
    trajectory(r, u0, p, (0, T); dt)

function trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, tspan::Tuple{T,T}; dt::T= one(T)) where {T<:Real}
  prob = DiscreteProblem(r, u0, tspan, p)
  solve(prob, FunctionMap(); dt = dt)
end

end