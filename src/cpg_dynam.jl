module CPortGraphDynam

using Catlab.WiringDiagrams
using Catlab.WiringDiagrams.CPortGraphs
using  Catlab.CategoricalAlgebra
using  Catlab.CategoricalAlgebra.FinSets
using Catlab.Graphs
import Catlab.Graphs: Graph
using Catlab.Graphics.GraphvizGraphs: to_graphviz

import Catlab.WiringDiagrams: oapply

import ..UWDDynam: nstates, nports, eval_dynamics, euler_approx
import ..DWDDynam: AbstractMachine, ContinuousMachine, DiscreteMachine, 
ninputs, noutputs
import Catlab.CategoricalAlgebra.CSets: migrate!

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

colimitmap!(f::Function, output, C::Colimit, input) = begin
    for (i,x) in enumerate(input)
        y = f(i, x)
        I = legs(C)[i](1:length(y))
        # length(I) == length(y) || error("colimitmap! attempting to fill $(length(I)) slots with $(length(y)) values")
        output[I] .= y
    end
    return output
end

@inline fillreadouts!(y, d, xs, Ports, statefun) = colimitmap!(y, Ports, xs) do i,x
    return x.readout(statefun(i))
end

@inline fillstates!(y, d, xs, States, statefun, inputfun, p, t) = colimitmap!(y, States, xs) do i, x
    return x.dynamics(statefun(i), inputfun(i), p, t)
end

function oapply(d::OpenCPortGraph, x::AbstractMachine)
    oapply(d, collect(repeated(x, nparts(d, :Box))))
end

fillreadins!(readins, d, readouts) = begin
    for b in parts(d, :Box)
        ports = incident(d, b, :box)
        for p in ports
            ws = incident(d, p, :tgt) 
            qs = d[ws, :src]
            readins[p] += sum(readouts[qs])
        end
    end
    return readins
end


"""Implements the operad algebras CDS and DDS given a 
composition pattern (implemented by an open circular port graph)
and primitive systems (implemented by a collection of 
machines).

Each box of the open CPG is filled by a machine with the 
appropriate type signature. Returns the composite machine.
"""
function oapply(d::OpenCPortGraph, xs::Vector{Machine}) where {T, Machine<:AbstractMachine{T}}
    x -> FinSet(x.nstates)
    S = coproduct((FinSet∘nstates).(xs))
    Inputs = coproduct((FinSet∘ninputs).(xs))
    Ports = coproduct([FinSet.(nports(d, b)) for b in parts(d, :Box)])
    state(u::Vector, b::Int) = u[legs(S)[b](1:xs[b].nstates)]
    readouts = zeros(T, length(apex(Ports)))
    readins  = zeros(T, length(apex(Ports)))
    ϕ = zeros(T, length(apex(S)))
    v = (u::AbstractVector, x::AbstractVector, p, t::Real) -> begin
        # length(p) == length(d[:, :con]) || error("Expected $(length(d[:, :con])) parameters, have $(length(p))")
        statefun(b) = state(u,b)
        inputfun(b) = readins[incident(d, b, :box)]
        fillreadouts!(readouts, d, xs, Ports, statefun)
        # communicate readouts to the ports at the other end of the wires, external connections directly fill ports
        readins .= 0 
        fillreadins!(readins, d, readouts)
        readins[d[:, :con]] .+= x
        fillstates!(ϕ, d, xs, S, statefun, inputfun, p, t)
        return ϕ
    end
    function readout(u::AbstractVector)
        statefun(b) = state(u,b)
        fillreadouts!(readouts, d, xs, Ports, statefun)
        return readouts[d[:, :con]]
    end
    return Machine( nparts(d, :OuterPort), apex(S).set, v, readout)
end


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
