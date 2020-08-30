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

export dynam, vectorfield


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
    for box in 1:nparts(d, :Box)
        n = subpart(d, :name)[box]
        ports = incident(d, box, :box)
        juncs = [subpart(d,:junction)[p] for p in ports]
        tk = (u, θ, t) -> generators[n](u[juncs], θ, t)
        push!(tasks, tk)
    end
    # this function could avoid doing all the lookups every time by enclosing the ports
    # vectors into the function.
    # TODO: Eliminate all allocations here
    aggregate!(out, du) = for j in 1:nparts(d, :Junction)
        ports = incident(d, j, :junction)
        out[j] = sum(du[ports])
    end
    return tasks, aggregate!
end

"""    vectorfield(d, generators::Dict)

create a function for computing the vector field of a dynamical system

- d: An undirected wiring diagram whose Boxes represent systems and junctions represent variables
- generators: A dictionary mapping the name of each box to its corresponding dynamical system

returns f(du, u, p, t) that you can pass to ODEProblem.
"""
function vectorfield(d, generators::Dict)
    tasks, aggregate! = dynam(d, generators)
    f(du, u, p, t) = begin
        # TODO: Eliminate all allocations here
        tvecs = map(enumerate(tasks)) do (i, tk)
            tk(u, p[i], t)
        end
        tvec = vcat(tvecs...)
        # @assert (length(tvec)) == nparts(d,:Port)
        aggregate!(du, tvec)
        return du
    end
    return f
end
end #module
