using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam

using OrdinaryDiffEq

using Base.Iterators


"""    trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, nsteps::Int; dt::Int = 1)

Evolves the resouce sharer `r` for `nsteps` times with step size `dt`, initial condition `u0`, and parameters `p`.
"""
function trajectory(r::DiscreteResourceSharer, u0::AbstractVector, p, T::Int; dt::Int= 1)
  prob = DiscreteProblem(r, u0, (0, T), p)
  sol = solve(problem, FunctionMap(); dt = dt)
  return sol.xs
end


"""    trajectory(m::DiscreteMachine, u0::AbstractVector, xs::AbstractVector, p, nsteps::Int; dt::Int = 1)

Evolves the machine `m` for `nsteps` times with step size `dt`, initial condition `u0`, and parameters `p`. Any inputs to `m` are determied by `xs`. If `m` has no inputs then you can omit `xs`.
"""
function trajectory(m::DiscreteMachine, u0::AbstractVector, xs,  p, T::Int; dt::Int= 1) 
  prob = DiscreteProblem(m, u0, xs, (0, T), p)
  sol = solve(prob, FunctionMap(); dt = dt)
  return sol.xs
end
