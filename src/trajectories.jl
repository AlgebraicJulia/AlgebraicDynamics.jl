using AlgebraicDynamics.UWDDynam
using AlgebraicDynamics.DWDDynam
using AlgebraicDynamics.DWDDynam: get_inputs

using StaticArrays
using DynamicalSystems
import DynamicalSystems: trajectory

using Base.Iterators

function trajectory1d(update::Function, u0::Vector{S}, T::Int, p; dt::Int) where S
  dt = round(Int, dt)
  tvec = 0:dt:T
  L = length(tvec)
  data = Vector{S}(undef, L)
  data[1] = u0[1]
  u = copy(u0)
  for i in 2:L
    for j in 1:dt
      u = update(u, p, (i - 1)*dt + j)
    end
    data[i] = u[1]
  end
  return data
end

function trajectory(update::Function, u0::Vector{S}, T::Int, p; dt::Int) where S
  if length(u0) == 1 
    return trajectory1d(update, u0, T, p; dt = dt)
  end
  dt = round(Int, dt)
  tvec = 0:dt:T
  L = length(tvec)
  data = Vector{SVector{length(u0), S}}(undef, L)
  data[1] = u0
  u = copy(u0)
  for i in 2:L
    for j in 1:dt
      u = update(u, p, (i - 1)*dt + j)
    end
    data[i] = u
  end
  return Dataset(data)
end

trajectory(r::DiscreteResourceSharer, u0::Vector, T::Int, p; dt = 1) = 
  trajectory((u,p,t) -> eval_dynamics(r, u, p, t), u0, T, p; dt = dt)

trajectory(m::DiscreteMachine, fs::Vector, u0::Vector, T::Int, p; dt = 1) = 
  trajectory((u,p,t) -> eval_dynamics(m, u, get_inputs(fs, t), p, t), u0, T, p; dt = dt)

trajectory(m::DiscreteMachine, f::Function, u0::Vector, T::Int, p; dt = 1) = 
  trajectory(m, collect(repeated(f, ninputs(m))), u0, T, p; dt = dt)
