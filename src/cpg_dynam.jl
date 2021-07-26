module CPortGraphDynam

using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
using Catlab.Graphs
import Catlab.Graphs: Graph
using Catlab.Graphics.GraphvizGraphs: to_graphviz

import Catlab.WiringDiagrams: oapply

using ..DWDDynam
using ...DWDDynam: destruct
import ..UWDDynam: nstates, nports, eval_dynamics, euler_approx
import ..DWDDynam: AbstractMachine, ContinuousMachine, DiscreteMachine, 
ninputs, noutputs
import Catlab.CategoricalAlgebra: migrate!

using Base.Iterators
import Base: show, eltype


ContinuousMachine{T}(nports::Int, nstates::Int, d::Function, r::Function) where T =
    ContinuousMachine{T}(nports, nstates, nports, d, r)
DiscreteMachine{T}(nports::Int, nstates::Int, d::Function, r::Function) where T =
    DiscreteMachine{T}(nports, nstates, nports, d, r)

migrate!(g::Graph, pg::OpenCPortGraph) = migrate!(g, migrate!(CPortGraph(), pg))

draw(g::Graph) = to_graphviz(g, prog="neato", edge_labels=true, node_labels=true)
draw(pg::OpenCPortGraph) = draw(migrate!(Graph(), pg))

concat(xs::Vector) = (collect ∘ Iterators.flatten)(xs)

nports(d::OpenCPortGraph, b::Int) = incident(d, b, :box) |> length
nports(d::OpenCPortGraph, b) = map(length, incident(d, b, :box))
nports(d::OpenCPortGraph, b::Colon) = map(length, incident(d, :, :box))

function fills(m::AbstractMachine, d::OpenCPortGraph, b::Int)
    nports = length(incident(d, b, :box))
    return (nports == ninputs(m)) && (nports == noutputs(m))
end

"""    oapply(d::OpenCPortGraph, ms::Vector)

Implements the operad algebras for directed composition of dynamical systems given a 
composition pattern (implemented by an open circular port graph `d`)
and primitive systems (implemented by a collection of 
machines `ms`).

Each box of the composition pattern `d` is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::OpenCPortGraph, ms::Vector{Machine}) where {T, Machine<:AbstractMachine{T}}
    @assert nparts(d, :Box) == length(ms)
    for b in 1:nparts(d, :Box)
        @assert fills(ms[b], d, b)
    end

    S = coproduct((FinSet∘nstates).(ms))

    function v(u::AbstractVector, xs::AbstractVector, p, t::Real)
        states = destruct(S, u)
        readins = zeros(T, nparts(d, :Port)) # in port order 

        for (b,m) in enumerate(ms)
            readouts = readout(m, states[b], p, t)
            for (i, port) in enumerate(incident(d, b, :box))
                for w in incident(d, port, :src)
                    readins[subpart(d, w, :tgt)] += readouts[i]
                end
            end
        end

        for (i,x) in enumerate(xs)
            readins[subpart(d, i, :con)] += x
        end

        return reduce(vcat, map(enumerate(ms)) do (b,m)
            eval_dynamics(m, collect(states[b]), view(readins, incident(d, b, :box)), p, t)
        end)
    end

    function r(u::AbstractVector, p, t::Real)
        states = destruct(S, u)
        port_readout = zeros(T, nparts(d, :Port))

        for (b,m) in enumerate(ms)
            readouts = readout(m, states[b], p, t)
            for (i, port) in enumerate(incident(d, b, :box))
                port_readout[port] = readouts[i]
            end
        end
        
        return collect(view(port_readout, subpart(d, :con)))
    end
    
    return Machine(nparts(d, :OuterPort), length(apex(S)), v, r)
end

"""    oapply(d::OpenCPortGraph, m::AbstractMachine)

A version of `oapply` where each box of `d` is filled with the machine `m`.
"""
function oapply(d::OpenCPortGraph, x::AbstractMachine)
    oapply(d, collect(repeated(x, nparts(d, :Box))))
end

"""    barbell(n::Int)

Constructs an open CPG with two boxes each with `n` ports. The ``i``th ports on each box are connected.
"""
barbell(k::Int) = begin
  g = OpenCPortGraph()
  add_parts!(g, :Box, 2)
  add_parts!(g, :Port, 2k; box=[fill(1,k); fill(2,k)])
  add_parts!(g, :Wire, k; src=1:k, tgt=k+1:2k)
  add_parts!(g, :Wire, k; tgt=1:k, src=k+1:2k)
  return g
end

meshpath(n::Int) = begin
    gt = @acset OpenCPortGraph begin
        Box = 1
        Port = 3
        Wire = 0
        OuterPort = 2
        box= ones(Int, 3)
        con= [3,2]
    end
    gm = @acset OpenCPortGraph begin
        Box = 1
        Port = 4
        Wire = 0
        OuterPort = 2
        box= ones(Int, 4)
        con= [4,2]
    end
    subs = [gt]
    for i in 2:n-1
        push!(subs, gm)
    end
    push!(subs, gt)
    X = coproduct(subs)
    for i in 1:n-1
        xi = subs[i]
        xj = subs[i+1]
        p = legs(X)[i][:Port](nparts(xi, :Port)-1)
        q = legs(X)[i+1][:Port](1)
        add_parts!(apex(X), :Wire, 2, src=[p,q], tgt=[q,p])
    end
    c₁ = apex(X)[1:2:nparts(apex(X),:OuterPort) ,:con]
    c₂ = apex(X)[2:2:nparts(apex(X),:OuterPort) ,:con]
    apex(X)[:,:con] = vcat(c₁,c₂)
    return X
end

function gridpath(n::Int, width::Int)
    node = @acset OpenCPortGraph begin
        Box = 1
        Port = 0
        Wire = 0
        box = 1
    end
    add_parts!(node, :Port, 2width, box=1)
    X = coproduct(collect(repeated(node, n)))
    L = legs(X)
    A = apex(X)
    for i in 1:n-1
        for j in 1:width
            s = L[i][:Port](j)
            t = L[i+1][:Port](j+width)
            add_part!(A, :Wire, src=s, tgt=t)
            add_part!(A, :Wire, src=t, tgt=s)
        end
    end
    upstream = L[1][:Port](width+1:2width)
    add_parts!(A, :OuterPort, width, con=upstream)
    downstream = L[end][:Port](1:width)
    add_parts!(A, :OuterPort, width, con=downstream)
    return X
end

grid(n::Int, m::Int) = ocompose(apex(gridpath(n,m)), collect(repeated(apex(meshpath(m)), n)))

end
