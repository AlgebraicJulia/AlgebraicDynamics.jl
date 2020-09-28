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

export dynam, dynam!, vectorfield, vectorfield!


"""    dynam(d, generators::Dict)

create a function for computing the dynamics of a wiring diagram. Returns a list of independent tasks
for each subsystem and a function for aggregating the terms of the ODE.

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system
"""
function dynam(d, generators::Dict)
    # we create a task array so that you could parallelize the computation of
    # primitive subsystems using pmap. This function overhead could be eliminated
    # for benchmarking the sequential version.
    tasks = Function[]

    cur_size = nparts(d, :Junction)
    num_juncs = nparts(d, :Junction)
    junctions = subpart(d,:junction)
    names = subpart(d, :name)

    # First we create a function based on each generator that accepts the total
    # state of the system (which is why it limits the input to `juncs`)
    for box in 1:nparts(d, :Box)
        n = names[box]
        ports = incident(d, box, :box)
        juncs = [junctions[p] for p in ports]
        (generator, size) = generators[n]
        if size > length(ports)
            len_diff = size - length(ports)
            append!(juncs, cur_size .+ (1:len_diff))
            cur_size += len_diff
        end
        tk = (u, θ, t) -> generator(u[juncs], θ, t)
        push!(tasks, tk)
    end
    # this function could avoid doing all the lookups every time by enclosing the ports
    # vectors into the function.
    # TODO: Eliminate all allocations here

    # Given a junction, what are all incident ports?
    inc = [incident(d,j,:junction) for j in 1:nparts(d, :Junction)]

    hidden_size = cur_size - num_juncs
    # Sum over all port values for each junction
    aggregate!(out, du) = begin
      for j in 1:length(inc)
        ports = inc[j]
        out[j] = sum(du[ports])
      end
      out[(end - hidden_size):end] = du[(end - hidden_size):end]
    end
    return tasks, aggregate!, cur_size
end

"""    vectorfield(d, generators::Dict)

create a function for computing the vector field of a dynamical system

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system

returns f(du, u, p, t) that you can pass to ODEProblem.
"""
function vectorfield(d, generators::Dict)

    tasks, aggregate!, num_var = dynam(d, generators)
    f(du, u, p, t) = begin
        # TODO: Eliminate all allocations here
        # Evaluate every individual generator
        tvecs = map(enumerate(tasks)) do (i, tk)
            tk(u, p[i], t)
        end

        # Since ports are assigned in order of Boxes, we can just concatenate
        # all tvecs and tvecs[n] will be the value at port `n`
        tvec = vcat(tvecs...)
        # @assert (length(tvec)) == nparts(d,:Port)
        aggregate!(du, tvec)
        return du
    end
    return f, num_var
end

"""    dynam!(d, generators::Dict, scratch::AbstractVector)

in place version of dynam

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system
- scratch::AbstractVector the storage space for computing the tangent vector, must have the same length as the number of ports
"""
function dynam!(d, generators::Dict, scratch::AbstractVector)
    juncs = subpart(d, :junction)
    names = subpart(d,:name)
    cur_ports = nparts(d, :Port)
    cur_juncs = nparts(d, :Junction)
    junc_size = nparts(d, :Junction)

    tasks = map(1:nparts(d, :Box)) do b
        boxports = copy(incident(d, b, :box))
        boxjuncs = juncs[boxports]
        n = names[b]
        (generator, size) = generators[n]
        if size > length(boxports)
            len_diff = size - length(boxports)
            append!(boxjuncs, cur_juncs .+ (1:len_diff))
            append!(boxports, cur_ports .+ (1:len_diff))
            cur_ports += len_diff
            cur_juncs += len_diff
        end
        tv = view(scratch, boxports)
        task(u, p, t) = begin
            v = view(u, boxjuncs)
            generator(tv, v, p, t)
        end
    end

    # TODO: Eliminate all allocations here
    inc = [incident(d,j, :junction) for j in 1:nparts(d,:Junction)]
    hidden_size = cur_juncs - junc_size
    aggregate!(out, du) = begin
        for j in 1:length(inc)
            juncports = inc[j]
            out[j] = sum(du[juncports])
        end
        out[(end - hidden_size):end] = du[(end - hidden_size):end]
    end
    @assert length(scratch) == cur_ports
    return tasks, aggregate!, cur_juncs
end

"""    vectorfield!(d, generators::Dict)

in place version of vectorfield

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system
- scratch::AbstractVector the storage space for computing the tangent vector, must have the same length as the number of ports

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
function vectorfield!(d, generators::Dict, scratch::AbstractVector)
    tasks, aggregate!, num_vars = dynam!(d,generators, scratch)
    f(du, u, p, t) = begin
        #TODO: parallelize this loop
        map(enumerate(tasks)) do (i, tk)
            tk(u, p[i], t)
        end
        # @assert (length(tvec)) == nparts(d,:Port)
        aggregate!(du, scratch)
        return du
    end
    return f, num_vars
end
end #module
