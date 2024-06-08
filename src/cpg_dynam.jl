module CPortGraphDynam

using Catlab

import Catlab.Graphs: Graph
using Catlab.Graphics.GraphvizGraphs: to_graphviz

import Catlab.WiringDiagrams: oapply

using ..DWDDynam
using ...DWDDynam: AbstractInterface, destruct, get_readouts
import ..UWDDynam: nstates, nports, eval_dynamics, euler_approx, fills
import ..DWDDynam: AbstractMachine, ContinuousMachine, DiscreteMachine, DelayMachine,
ninputs, noutputs
import Catlab.CategoricalAlgebra: migrate!

using Base.Iterators


ContinuousMachine{T, I}(nports::Int, nstates::Int, d::Function, r::Function) where {T,I} =
    ContinuousMachine{T, I}(nports, nstates, nports, d, r)
ContinuousMachine{T}(nports::Int, nstates::Int, d::Function, r::Function) where T =
    ContinuousMachine{T}(nports, nstates, nports, d, r)
DiscreteMachine{T, I}(nports::Int, nstates::Int, d::Function, r::Function) where {T,I} =
    DiscreteMachine{T, I}(nports, nstates, nports, d, r)
DiscreteMachine{T}(nports::Int, nstates::Int, d::Function, r::Function) where T =
    DiscreteMachine{T}(nports, nstates, nports, d, r)
DelayMachine{T, I}(nports::Int, nstates::Int, d::Function, r::Function) where {T,I} =
    DelayMachine{T, I}(nports, nstates, nports, d, r)
DelayMachine{T}(nports::Int, nstates::Int, d::Function, r::Function) where T =
    DelayMachine{T}(nports, nstates, nports, d, r)

migrate!(g::Graph, pg::OpenCPortGraph) = migrate!(g, migrate!(CPortGraph(), pg))

draw(g::Graph) = to_graphviz(g, prog="neato", edge_labels=true, node_labels=true)
draw(pg::OpenCPortGraph) = draw(migrate!(Graph(), pg))

concat(xs::Vector) = (collect ∘ Iterators.flatten)(xs)

nports(d::OpenCPortGraph, b::Int) = incident(d, b, :box) |> length
nports(d::OpenCPortGraph, b) = map(length, incident(d, b, :box))
nports(d::OpenCPortGraph, b::Colon) = map(length, incident(d, :, :box))

"""    fills(m::AbstractMachine, d::OpenCPortGraph, b::Int)

Checks if `m` is of the correct signature to fill box `b` of the open CPG `d`.
"""
function fills(m::AbstractMachine, d::OpenCPortGraph, b::Int)
    nports = length(incident(d, b, :box))
    return (nports == ninputs(m)) && (nports == noutputs(m))
end

"""    oapply(d::OpenCPortGraph, ms::Vector{M}) where {M<:AbstractMachine}

Implements the operad algebras for directed composition of dynamical systems, given a
composition pattern (implemented by an open circular port graph `d`)
and primitive systems (implemented by a collection of 
machines `ms`). Returns the composite machine.

Each box of the composition pattern `d` must be filled by a machine with the
appropriate type signature.
"""
function oapply(d::OpenCPortGraph, ms::Vector{M}) where {M<:AbstractMachine}
    @assert nparts(d, :Box) == length(ms)
    for b in 1:nparts(d, :Box)
        @assert fills(ms[b], d, b)
    end

    S = coproduct((FinSet∘nstates).(ms))
    
    return M(
        nparts(d, :OuterPort), 
        length(apex(S)), 
        induced_dynamics(d, ms, S), 
        induced_readout(d, ms, S)
    )
end


"""    oapply(d::OpenCPortGraph, m::AbstractMachine)

A version of `oapply` where each box of `d` is filled with the same machine `m`.
"""
function oapply(d::OpenCPortGraph, x::AbstractMachine)
    oapply(d, collect(repeated(x, nparts(d, :Box))))
end


function induced_dynamics(d::OpenCPortGraph, ms::Vector{M}, S) where {T, I, M<:AbstractMachine{T, I}}

    function v(u::AbstractVector, xs::AbstractVector, p, t)
        states = destruct(S, u)
        port_readouts = get_port_readout(d, ms, states, p, t)
          
        reduce(vcat, map(parts(d, :Box)) do i 
          inputs = map(incident(d, i, :box)) do port
            sum(map(incident(d, port, :tgt)) do w 
              port_readouts[d[:src][w]]
            end; init = sum(xs[incident(d, port, :con)]; init = zero(I)))
          end

          eval_dynamics(ms[i], collect(states[i]), inputs, p, t)
        end)
      end
    end



function induced_dynamics(d::OpenCPortGraph, ms::Vector{M}, S) where {T, I, M<:DelayMachine{T, I}}
    function v(u::AbstractVector, xs::AbstractVector, h, p, t::Real)
        states = destruct(S, u)
        hists = destruct(S, h)
        port_readouts = get_port_readout(d, ms, states, hists, p, t)
          
        reduce(vcat, map(parts(d, :Box)) do i 
          inputs = map(incident(d, i, :box)) do port
            sum(map(incident(d, port, :tgt)) do w 
              port_readouts[d[:src][w]]
            end; init = sum(xs[incident(d, port, :con)]; init = zero(I)))
          end

          eval_dynamics(ms[i], collect(states[i]), inputs, hists[i], p, t)
        end)
    end
end

function induced_readout(d::OpenCPortGraph, ms::Vector{M}, S) where {T, I, M<:AbstractMachine{T, I}}
    function r(u::AbstractVector, p, t::Real)
        states = destruct(S, u)
        port_readout = get_port_readout(d, ms, states, p, t)
        return collect(view(port_readout, subpart(d, :con)))
    end
end

function induced_readout(d::OpenCPortGraph, ms::Vector{M}, S) where {T, I, M<:DelayMachine{T, I}}
    function r(u::AbstractVector, h::Function, p, t::Real)
        states = destruct(S, u)
        hists = destruct(S, h)
        port_readout = get_port_readout(d, ms, states, hists, p, t)
        return collect(view(port_readout, subpart(d, :con)))
    end
end

function get_port_readout(d::OpenCPortGraph, ms::Vector{M}, states, args...) where M <: AbstractMachine
  readouts = get_readouts(ms, states, args...)
  map(parts(d, :Port)) do port 
    b = d[:box][port]
    idx = findfirst(isequal(port), incident(d, b, :box))
    readouts[b][idx]
  end
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
