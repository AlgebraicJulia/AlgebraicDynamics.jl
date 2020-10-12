module Hypergraphs

# import Base: +, sum, inv
using LinearAlgebra
using Catlab
# import Catlab.Theories:
#   Ob, Hom, dom, codom, compose, ⋅, ∘, id, oplus, ⊕, mzero, swap,
#   dagger, dunit, dcounit, mcopy, Δ, delete, ◊, mmerge, ∇, create, □,
#   plus, +, zero, coplus, cozero, meet, top, join, bottom,
#   proj1, proj2, pair, copair
using Catlab.LinearAlgebra
import Catlab.LinearAlgebra: scalar, antipode

using Catlab.Programs.RelationalPrograms
using Catlab.CategoricalAlgebra
using Catlab.CategoricalAlgebra.CSets

export DynamicalSystem, dynam, vectorfield, set_junctions!, set_hidden_vars!, State


struct State
  state::Array{Float64,1}
  hidden_ports::Array{Int,1}
  visible_ports::Array{Int,1}
  ports_to_junctions::Array{Int,1}
end

struct DynamicalSystem
  dynamics::F where F <: Function
  space::Int
end

function set_junctions!(state::State, values::Array{<:Real, 1})
  maximum(state.ports_to_junctions) == length(values) || throw(DimensionMismatch("Not enough ports for input values"))
  state.state[state.visible_ports] = values[state.ports_to_junctions]
end

function set_hidden_vars!(state::State, values::Array{<:Real, 1})
  size(state.hidden_ports) == size(values) || throw(DimensionMismatch("Not enough ports for input values"))
  state.state[state.hidden_ports] = values
end

"""    dynam!(d, generators::Dict, scratch::AbstractVector)

in place version of dynam

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system
- scratch::AbstractVector the storage space for computing the tangent vector, must have the same length as the number of ports
"""
function dynam(d, generators::Dict{Symbol, DynamicalSystem})
    juncs = subpart(d, :junction)
    names = subpart(d,:name)
    cur_ports = nparts(d, :Port)
    cur_juncs = nparts(d, :Junction)
    junc_size = nparts(d, :Junction)

    cur_state = 0
    offset = 0
    port_to_state = zeros(Int, cur_ports)

    hidden_states = []

    # Define the task list
    tasks = map(1:nparts(d, :Box)) do b
        boxports = collect(incident(d, b, :box))
        state_vals = offset .+ boxports
        n = names[b]
        dynam = generators[n].dynamics
        space = generators[n].space

        port_to_state[boxports] = state_vals
        cur_state += length(boxports)

        if space > length(boxports)
            len_diff = space - length(boxports)
            append!(state_vals, cur_state .+ (1:len_diff))
            append!(hidden_states, cur_state .+ (1:len_diff)) 
            cur_state += len_diff
            offset += len_diff
        end

        # Run dynam with only its own ports visible
        task(du, u, p, t) = begin
            v = view(u, state_vals)
            tv = view(du, state_vals)
            dynam(tv, v, p, t)
        end
    end

    # Define the aggregation function
    #inc = [incident(d,j, :junction) for j in 1:nparts(d,:Junction)]
    junc_state = zeros(junc_size)
    port_range = 1:nparts(d, :Port)

    p2j = subpart(d, :junction)
    j2p = [incident(d,j, :junction)[1] for j in 1:nparts(d, :Junction)]
    internal_junc = port_to_state[j2p[p2j]]

    aggregate!(du) = begin
        junc_state .= 0
        for i in port_range
            if internal_junc[i] != port_to_state[i]
                du[internal_junc[i]] += du[port_to_state[i]]
            end
        end
        du[port_to_state[port_range]] = du[internal_junc[port_range]]
    end
    return tasks, aggregate!, State(zeros(cur_state), hidden_states, port_to_state, p2j)
end

"""    vectorfield!(d, generators::Dict)

in place version of vectorfield

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system

returns f(du, u, p, t) that you can pass to ODEProblem.

The generators passed to this function must accept `du::AbstractVector` passed as the first argument

Example:

```julia
    d = @relation (x,y) where (x::X, y::X) begin
        birth(x)
        predation(x,y)
        death(y)
    end

    α = 1.2
    β = 0.1
    γ = 1.3
    δ = 0.1

    g = Dict(
        :birth     => (du, u, p, t) -> begin du[1] =  α*u[1] end,
        :death     => (du, u, p, t) -> begin du[1] = -γ*u[1] end,
        :predation => (du, u, p, t) -> begin du[1] = -β*u[1]*u[2]
        du[2] = δ*u[1]*u[2]
        end
    )
    vectorfield!(d, g, zeros(3))
```
"""
function vectorfield(d, generators::Dict)
    tasks, aggregate!, state = dynam(d,generators)
    f(du, u, p, t) = begin
        #TODO: parallelize this loop
        map(enumerate(tasks)) do (i, tk)
            tk(du, u, p[i], t)
        end
        # @assert (length(tvec)) == nparts(d,:Port)
        aggregate!(du)
        return du
    end
    return f, state
end
end #module
